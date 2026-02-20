from pxr import Usd, Sdf, UsdGeom
import os

# 경로 설정
src_path = "/home/Isaac/workspace/HD/Real2Sim/isaacsim_assets_for_export/Collected_World0/World0.usd"
dst_path = "/home/Isaac/workspace/HD/Real2Sim/isaacsim_assets_for_export/ur5e_tesollo.usd"

# 기존 파일 삭제
if os.path.exists(dst_path):
    os.remove(dst_path)

# Stage 열기
stage = Usd.Stage.Open(src_path)

# 1. 먼저 flatten하여 직접 저장 (모든 참조와 prototype 포함)
print("Flatten 후 저장 중...")
flattened_layer = stage.Flatten()
flattened_layer.Export(dst_path)
print(f"Flattened USD 저장: {dst_path}")

# 2. 저장된 파일 다시 열기
edit_stage = Usd.Stage.Open(dst_path)

# 3. 불필요한 prim 삭제
delete_paths = [
    # ur5e 내부의 불필요한 것들
    "/World/simulation_env/ur5e/Gripper",
    "/World/simulation_env/ur5e/Merged_Robot_flange",
    # ur5e 외부의 것들
    "/World/simulation_env/work_bench",
    "/World/simulation_env/OpticalBreadBoard",
    "/World/simulation_env/GroundPlane",
    "/World/simulation_env/Camera",
    "/World/simulation_env/Camera_01",
    "/World/simulation_env/CollisionGroup",
    "/Environment",
    "/Render",
]

for path in delete_paths:
    prim = edit_stage.GetPrimAtPath(path)
    if prim:
        edit_stage.RemovePrim(path)
        print(f"삭제: {path}")

# 4. ur5e를 루트로 이동 (Sdf 레벨에서 rename)
edit_layer = edit_stage.GetRootLayer()

# ur5e의 새 이름 설정
old_ur5e_path = Sdf.Path("/World/simulation_env/ur5e")
new_ur5e_path = Sdf.Path("/ur5e_tesollo")

# Sdf.CopySpec으로 ur5e를 루트로 복사
Sdf.CopySpec(edit_layer, old_ur5e_path, edit_layer, new_ur5e_path)
print(f"ur5e를 {new_ur5e_path}로 복사")

# 기존 World 삭제
edit_stage.RemovePrim("/World")
print("기존 /World 삭제")

# 5. defaultPrim 설정
ur5e_prim = edit_stage.GetPrimAtPath("/ur5e_tesollo")
if ur5e_prim:
    edit_stage.SetDefaultPrim(ur5e_prim)
    print("defaultPrim 설정: /ur5e_tesollo")

# 6. Material binding 복구
from pxr import UsdShade

print("\nMaterial binding 복구 중...")

# UR5e materials 매핑 (mesh 이름 패턴 -> material)
ur5e_material_map = {
    "base_link": "/ur5e_tesollo/Looks/LinkGrey",
    "shoulder": "/ur5e_tesollo/Looks/JointGrey",
    "upper_arm": "/ur5e_tesollo/Looks/LinkGrey",
    "forearm": "/ur5e_tesollo/Looks/LinkGrey",
    "wrist_1": "/ur5e_tesollo/Looks/JointGrey",
    "wrist_2": "/ur5e_tesollo/Looks/JointGrey",
    "wrist_3": "/ur5e_tesollo/Looks/JointGrey",
}

# Tesollo hand material
tesollo_material = "/ur5e_tesollo/dg5f_L_final/dg5f_left_flattened/Looks/Material_001"

binding_count = 0
for prim in edit_stage.Traverse():
    if prim.GetTypeName() == "Mesh":
        path_str = str(prim.GetPath())
        mat_path = None

        # UR5e 메시 확인
        for key, mat in ur5e_material_map.items():
            if key in path_str and "dg5f" not in path_str:
                mat_path = mat
                break

        # Tesollo hand 메시
        if "dg5f" in path_str and mat_path is None:
            mat_path = tesollo_material

        # Material binding 설정
        if mat_path:
            mat_prim = edit_stage.GetPrimAtPath(mat_path)
            if mat_prim:
                material = UsdShade.Material(mat_prim)
                binding = UsdShade.MaterialBindingAPI.Apply(prim)
                binding.Bind(material)
                binding_count += 1

print(f"Material binding 복구: {binding_count}개")

# 7. 저장
edit_stage.GetRootLayer().Save()
print(f"\n저장 완료: {dst_path}")

# 검증
print("\n=== 최종 구조 (ur5e_tesollo 하위) ===")
verify_stage = Usd.Stage.Open(dst_path)
count = 0
for prim in verify_stage.Traverse():
    if prim.GetParent().GetPath() == Sdf.Path("/ur5e_tesollo"):
        print(f"  {prim.GetName()}")
        count += 1
print(f"\n총 {count}개 하위 prim")
