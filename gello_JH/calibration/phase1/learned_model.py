# Copyright (c) 2025, SRBL
# Phase 1: Learning-Based Current-Torque Calibration Model

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LearnedModelConfig:
    """Configuration for learned calibration model."""

    # Model architecture
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.1
    use_batch_norm: bool = True

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 200
    patience: int = 20  # Early stopping patience

    # Data
    val_split: float = 0.2
    normalize_inputs: bool = True

    # Input feature selection
    use_joint_pos: bool = True      # q (20)
    use_joint_vel: bool = True      # qdot (20)
    use_finger_ft: bool = True      # Tesollo FT per finger (6)
    use_real_current: bool = True   # I_real (20)

    # Output
    num_joints: int = 20


class TorqueEstimatorMLP(nn.Module):
    """MLP for τ_sim estimation from high-dimensional features.

    Input features: (q, qdot, FingerFT(6), I_real)
    Output: τ_sim (20)
    """

    def __init__(self, input_dim: int, output_dim: int, cfg: LearnedModelConfig):
        super().__init__()

        self.cfg = cfg
        layers = []
        prev_dim = input_dim

        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if cfg.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualTorqueEstimator(nn.Module):
    """Residual learning model: learns correction on top of linear k_t.

    τ_sim = k_t × I_real + f_residual(q, qdot, FT, I_real)

    The residual network learns non-linearities that the linear model misses
    (configuration-dependent effects, friction non-linearities, etc.)
    """

    def __init__(self, input_dim: int, output_dim: int, cfg: LearnedModelConfig):
        super().__init__()

        self.cfg = cfg
        self.output_dim = output_dim

        # Linear component (initialized from regression)
        self.k_t = nn.Parameter(torch.ones(output_dim))
        self.offset = nn.Parameter(torch.zeros(output_dim))

        # Residual network
        layers = []
        prev_dim = input_dim

        for hidden_dim in cfg.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if cfg.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.residual_net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, I_real: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Full feature vector (batch, input_dim).
            I_real: Real motor currents (batch, 20) — extracted for linear component.
        """
        linear = self.k_t * I_real + self.offset
        residual = self.residual_net(x)
        return linear + residual


class LearnedCurrentTorqueModel:
    """Learning-based current-to-torque calibration model.

    Uses high-dimensional features (q, qdot, FingerFT, I_real) to estimate τ_sim.
    Supports two architectures:
        1. MLP: Direct mapping features → τ_sim
        2. Residual: k_t × I + f_residual(features) → τ_sim

    The learning-based approach can capture:
        - Configuration-dependent gravity/friction
        - Non-linear motor characteristics
        - FT sensor cross-coupling effects
    """

    def __init__(self, cfg: LearnedModelConfig | None = None):
        self.cfg = cfg or LearnedModelConfig()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Models (created during training)
        self.mlp_model: TorqueEstimatorMLP | None = None
        self.residual_model: ResidualTorqueEstimator | None = None

        # Normalization stats
        self.input_mean: torch.Tensor | None = None
        self.input_std: torch.Tensor | None = None
        self.output_mean: torch.Tensor | None = None
        self.output_std: torch.Tensor | None = None

        # Training history
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        # Feature dimensions
        self._input_dim = 0
        self._current_start_idx = 0  # Index where I_real starts in feature vector

        # Finger info
        self.finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        self.finger_joint_indices = {
            "thumb": [0, 1, 2, 3],
            "index": [4, 5, 6, 7],
            "middle": [8, 9, 10, 11],
            "ring": [12, 13, 14, 15],
            "pinky": [16, 17, 18, 19],
        }

    def _compute_input_dim(self) -> int:
        """Compute input dimension based on config."""
        dim = 0
        if self.cfg.use_joint_pos:
            dim += 20
        if self.cfg.use_joint_vel:
            dim += 20
        if self.cfg.use_finger_ft:
            dim += 30  # 5 fingers × 6D
        if self.cfg.use_real_current:
            self._current_start_idx = dim
            dim += 20
        return dim

    def _build_features(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        F_internal: np.ndarray | None,
        I_real: np.ndarray,
    ) -> np.ndarray:
        """Build feature vector from raw data.

        Args:
            q: Joint positions (N, 20).
            qdot: Joint velocities (N, 20).
            F_internal: Internal FT sensor readings (N, 30), optional.
            I_real: Motor currents (N, 20).

        Returns:
            Feature matrix (N, input_dim).
        """
        features = []
        if self.cfg.use_joint_pos:
            features.append(q)
        if self.cfg.use_joint_vel:
            features.append(qdot)
        if self.cfg.use_finger_ft:
            if F_internal is not None:
                features.append(F_internal)
            else:
                features.append(np.zeros((len(q), 30)))
        if self.cfg.use_real_current:
            features.append(I_real)

        return np.hstack(features)

    def train_mlp(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        F_internal: np.ndarray | None,
        I_real: np.ndarray,
        tau_sim: np.ndarray,
    ) -> dict:
        """Train MLP model.

        Args:
            q: Joint positions (N, 20).
            qdot: Joint velocities (N, 20).
            F_internal: Internal FT sensor (N, 30), optional.
            I_real: Motor currents (N, 20).
            tau_sim: Target sim torques (N, 20).

        Returns:
            Training metrics dict.
        """
        self._input_dim = self._compute_input_dim()
        X = self._build_features(q, qdot, F_internal, I_real)
        y = tau_sim

        # Create model
        self.mlp_model = TorqueEstimatorMLP(
            self._input_dim, self.cfg.num_joints, self.cfg,
        ).to(self.device)

        return self._train_model(self.mlp_model, X, y, model_type="mlp")

    def train_residual(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        F_internal: np.ndarray | None,
        I_real: np.ndarray,
        tau_sim: np.ndarray,
        k_t_init: np.ndarray | None = None,
        offset_init: np.ndarray | None = None,
    ) -> dict:
        """Train residual model (linear k_t + learned residual).

        Args:
            q: Joint positions (N, 20).
            qdot: Joint velocities (N, 20).
            F_internal: Internal FT sensor (N, 30), optional.
            I_real: Motor currents (N, 20).
            tau_sim: Target sim torques (N, 20).
            k_t_init: Initial k_t from linear regression, optional.
            offset_init: Initial offset from linear regression, optional.

        Returns:
            Training metrics dict.
        """
        self._input_dim = self._compute_input_dim()
        X = self._build_features(q, qdot, F_internal, I_real)
        y = tau_sim

        # Create model
        self.residual_model = ResidualTorqueEstimator(
            self._input_dim, self.cfg.num_joints, self.cfg,
        ).to(self.device)

        # Initialize with linear regression results
        if k_t_init is not None:
            self.residual_model.k_t.data = torch.from_numpy(k_t_init).float().to(self.device)
        if offset_init is not None:
            self.residual_model.offset.data = torch.from_numpy(offset_init).float().to(self.device)

        return self._train_model(
            self.residual_model, X, y, model_type="residual", I_real=I_real,
        )

    def _train_model(
        self,
        model: nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "mlp",
        I_real: np.ndarray | None = None,
    ) -> dict:
        """Common training loop.

        Returns training metrics.
        """
        N = len(X)

        # Train/val split
        indices = np.random.permutation(N)
        val_size = int(N * self.cfg.val_split)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Normalize
        if self.cfg.normalize_inputs:
            self.input_mean = torch.from_numpy(X_train.mean(axis=0)).float().to(self.device)
            self.input_std = torch.from_numpy(X_train.std(axis=0) + 1e-8).float().to(self.device)
            self.output_mean = torch.from_numpy(y_train.mean(axis=0)).float().to(self.device)
            self.output_std = torch.from_numpy(y_train.std(axis=0) + 1e-8).float().to(self.device)

        # Convert to tensors
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).float().to(self.device)
        X_val_t = torch.from_numpy(X_val).float().to(self.device)
        y_val_t = torch.from_numpy(y_val).float().to(self.device)

        if self.cfg.normalize_inputs:
            X_train_t = (X_train_t - self.input_mean) / self.input_std
            X_val_t = (X_val_t - self.input_mean) / self.input_std
            y_train_t = (y_train_t - self.output_mean) / self.output_std
            y_val_t = (y_val_t - self.output_mean) / self.output_std

        # I_real tensors for residual model
        I_train_t = I_val_t = None
        if model_type == "residual" and I_real is not None:
            I_train_t = torch.from_numpy(I_real[train_idx]).float().to(self.device)
            I_val_t = torch.from_numpy(I_real[val_idx]).float().to(self.device)

        # Dataset / DataLoader
        train_dataset = torch.utils.data.TensorDataset(
            X_train_t, y_train_t,
            *([I_train_t] if I_train_t is not None else []),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg.batch_size, shuffle=True,
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5,
        )

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.train_losses = []
        self.val_losses = []

        print(f"\nTraining {model_type} model...")
        print(f"  Input dim: {self._input_dim}, Training samples: {len(train_idx)}, "
              f"Val samples: {val_size}")

        for epoch in range(self.cfg.num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                if model_type == "residual":
                    x_batch, y_batch, I_batch = batch
                    pred = model(x_batch, I_batch)
                else:
                    x_batch, y_batch = batch
                    pred = model(x_batch)

                loss = nn.functional.mse_loss(pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(x_batch)

            train_loss = epoch_loss / len(train_idx)

            # Validation
            model.eval()
            with torch.no_grad():
                if model_type == "residual":
                    val_pred = model(X_val_t, I_val_t)
                else:
                    val_pred = model(X_val_t)
                val_loss = nn.functional.mse_loss(val_pred, y_val_t).item()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{self.cfg.num_epochs}: "
                      f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            if patience_counter >= self.cfg.patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        # Compute final metrics
        model.eval()
        with torch.no_grad():
            if model_type == "residual":
                val_pred = model(X_val_t, I_val_t)
            else:
                val_pred = model(X_val_t)

            if self.cfg.normalize_inputs:
                val_pred_orig = val_pred * self.output_std + self.output_mean
                y_val_orig = y_val_t * self.output_std + self.output_mean
            else:
                val_pred_orig = val_pred
                y_val_orig = y_val_t

            # Per-joint R²
            ss_res = ((y_val_orig - val_pred_orig) ** 2).mean(dim=0)
            ss_tot = ((y_val_orig - y_val_orig.mean(dim=0)) ** 2).mean(dim=0)
            r_squared = (1.0 - ss_res / (ss_tot + 1e-8)).cpu().numpy()
            rmse = torch.sqrt(ss_res).cpu().numpy()

        metrics = {
            "best_val_loss": best_val_loss,
            "final_train_loss": self.train_losses[-1],
            "r_squared_per_joint": r_squared.tolist(),
            "mean_r_squared": float(np.mean(r_squared)),
            "rmse_per_joint": rmse.tolist(),
            "mean_rmse": float(np.mean(rmse)),
            "num_epochs_trained": len(self.train_losses),
            "num_train_samples": len(train_idx),
            "num_val_samples": val_size,
        }

        print(f"\nTraining complete!")
        print(f"  Mean R²: {metrics['mean_r_squared']:.4f}")
        print(f"  Mean RMSE: {metrics['mean_rmse']:.6f}")

        # Per-finger summary
        for finger_name, joint_indices in self.finger_joint_indices.items():
            finger_r2 = np.mean([r_squared[j] for j in joint_indices])
            print(f"  {finger_name}: R²={finger_r2:.4f}")

        return metrics

    # =========================================================================
    # Inference
    # =========================================================================

    def predict(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        F_internal: np.ndarray | None,
        I_real: np.ndarray,
        model_type: str = "mlp",
    ) -> np.ndarray:
        """Predict τ_sim from features.

        Args:
            q: Joint positions (..., 20).
            qdot: Joint velocities (..., 20).
            F_internal: Internal FT sensor (..., 30), optional.
            I_real: Motor currents (..., 20).
            model_type: "mlp" or "residual".

        Returns:
            Predicted τ_sim, shape (..., 20).
        """
        model = self.mlp_model if model_type == "mlp" else self.residual_model
        if model is None:
            raise RuntimeError(f"No trained {model_type} model available.")

        # Handle batch dims
        original_shape = q.shape[:-1]
        q_flat = q.reshape(-1, 20)
        qdot_flat = qdot.reshape(-1, 20)
        I_flat = I_real.reshape(-1, 20)
        F_flat = F_internal.reshape(-1, 30) if F_internal is not None else None

        X = self._build_features(q_flat, qdot_flat, F_flat, I_flat)
        X_t = torch.from_numpy(X).float().to(self.device)

        if self.cfg.normalize_inputs and self.input_mean is not None:
            X_t = (X_t - self.input_mean) / self.input_std

        model.eval()
        with torch.no_grad():
            if model_type == "residual":
                I_t = torch.from_numpy(I_flat).float().to(self.device)
                pred = model(X_t, I_t)
            else:
                pred = model(X_t)

            if self.cfg.normalize_inputs and self.output_mean is not None:
                pred = pred * self.output_std + self.output_mean

        result = pred.cpu().numpy()
        return result.reshape(*original_shape, 20)

    # =========================================================================
    # I/O
    # =========================================================================

    def save(self, filepath: str | Path):
        """Save model to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "cfg": {
                "hidden_dims": self.cfg.hidden_dims,
                "dropout": self.cfg.dropout,
                "use_batch_norm": self.cfg.use_batch_norm,
                "num_joints": self.cfg.num_joints,
                "use_joint_pos": self.cfg.use_joint_pos,
                "use_joint_vel": self.cfg.use_joint_vel,
                "use_finger_ft": self.cfg.use_finger_ft,
                "use_real_current": self.cfg.use_real_current,
                "normalize_inputs": self.cfg.normalize_inputs,
            },
            "input_dim": self._input_dim,
            "current_start_idx": self._current_start_idx,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        if self.input_mean is not None:
            save_dict["input_mean"] = self.input_mean.cpu()
            save_dict["input_std"] = self.input_std.cpu()
            save_dict["output_mean"] = self.output_mean.cpu()
            save_dict["output_std"] = self.output_std.cpu()

        if self.mlp_model is not None:
            save_dict["mlp_state"] = self.mlp_model.state_dict()
        if self.residual_model is not None:
            save_dict["residual_state"] = self.residual_model.state_dict()

        torch.save(save_dict, filepath)
        print(f"Saved learned model to {filepath}")

    def load(self, filepath: str | Path):
        """Load model from file."""
        filepath = Path(filepath)
        save_dict = torch.load(filepath, map_location=self.device, weights_only=False)

        # Restore config
        cfg_dict = save_dict["cfg"]
        for key, value in cfg_dict.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

        self._input_dim = save_dict["input_dim"]
        self._current_start_idx = save_dict.get("current_start_idx", 0)
        self.train_losses = save_dict.get("train_losses", [])
        self.val_losses = save_dict.get("val_losses", [])

        if "input_mean" in save_dict:
            self.input_mean = save_dict["input_mean"].to(self.device)
            self.input_std = save_dict["input_std"].to(self.device)
            self.output_mean = save_dict["output_mean"].to(self.device)
            self.output_std = save_dict["output_std"].to(self.device)

        if "mlp_state" in save_dict:
            self.mlp_model = TorqueEstimatorMLP(
                self._input_dim, self.cfg.num_joints, self.cfg,
            ).to(self.device)
            self.mlp_model.load_state_dict(save_dict["mlp_state"])

        if "residual_state" in save_dict:
            self.residual_model = ResidualTorqueEstimator(
                self._input_dim, self.cfg.num_joints, self.cfg,
            ).to(self.device)
            self.residual_model.load_state_dict(save_dict["residual_state"])

        print(f"Loaded learned model from {filepath}")

    def to_yaml_dict(self) -> dict:
        """Convert model info to YAML-serializable dictionary."""
        result = {
            "method": "learned",
            "architecture": {
                "hidden_dims": self.cfg.hidden_dims,
                "input_dim": self._input_dim,
                "output_dim": self.cfg.num_joints,
                "dropout": self.cfg.dropout,
                "use_batch_norm": self.cfg.use_batch_norm,
            },
            "features": {
                "use_joint_pos": self.cfg.use_joint_pos,
                "use_joint_vel": self.cfg.use_joint_vel,
                "use_finger_ft": self.cfg.use_finger_ft,
                "use_real_current": self.cfg.use_real_current,
            },
            "training": {
                "num_epochs_trained": len(self.train_losses),
                "final_train_loss": self.train_losses[-1] if self.train_losses else None,
                "final_val_loss": self.val_losses[-1] if self.val_losses else None,
                "best_val_loss": min(self.val_losses) if self.val_losses else None,
            },
            "has_mlp": self.mlp_model is not None,
            "has_residual": self.residual_model is not None,
        }

        return result

    def summary(self) -> str:
        """Get model summary string."""
        lines = ["=" * 50]
        lines.append("Phase 1 Learning-Based Calibration Summary")
        lines.append("=" * 50)
        lines.append(f"Input dim: {self._input_dim}")
        lines.append(f"Hidden dims: {self.cfg.hidden_dims}")
        lines.append(f"Features: q={self.cfg.use_joint_pos}, qdot={self.cfg.use_joint_vel}, "
                      f"FT={self.cfg.use_finger_ft}, I={self.cfg.use_real_current}")

        if self.mlp_model is not None:
            n_params = sum(p.numel() for p in self.mlp_model.parameters())
            lines.append(f"\nMLP Model: {n_params} parameters")

        if self.residual_model is not None:
            n_params = sum(p.numel() for p in self.residual_model.parameters())
            lines.append(f"Residual Model: {n_params} parameters")
            lines.append(f"  k_t: {self.residual_model.k_t.data.cpu().numpy()}")

        if self.train_losses:
            lines.append(f"\nTraining: {len(self.train_losses)} epochs")
            lines.append(f"  Best val loss: {min(self.val_losses):.6f}")

        lines.append("=" * 50)
        return "\n".join(lines)
