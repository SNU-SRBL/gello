#!/bin/bash

# Load conda functionality
source ~/miniconda3/etc/profile.d/conda.sh  # Or your real conda install path

# Deactivate base if active
conda deactivate 2>/dev/null

# Activate your intended conda environment
conda activate guest

# Add project root to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Start tmux session
tmux new-session -d -s gello

# Pane 0: camera nodes
tmux send-keys -t gello:0 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate guest && export PYTHONPATH="$PYTHONPATH:$(pwd)" && python experiments/launch_camera_nodes.py --hostname 127.0.0.1' C-m

# Pane 1: robot node
tmux split-window -h -t gello:0
tmux send-keys -t gello:0.1 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate guest && export PYTHONPATH="$PYTHONPATH:$(pwd)" && python experiments/launch_nodes.py --robot ur --robot_ip 192.168.0.1' C-m

# Split vertically from the left half (pane 0)
tmux split-window -v -t gello:0
sleep 0.5
tmux send-keys -t gello:0.2 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate guest && export PYTHONPATH="$PYTHONPATH:$(pwd)" && python experiments/run_env.py --agent=gello --use_save_interface --fsr' C-m

# Attach to the tmux session
tmux attach -t gello
