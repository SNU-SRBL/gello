"""
USD 파일 색상 및 마찰계수 설정 스크립트
- cylinder.usdc, cylinder_case.usdc: 색상 (0.05, 0.05, 0.05), friction 0.5
- Tseollo_integration.usdc: 색상 (0.1, 0.1, 0.1)
"""
import os
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Gf, Sdf, Vt

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))


def set_display_color(stage, color):
    """모든 Mesh 프림에 displayColor 설정"""
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
            print(f"  색상 설정: {prim.GetPath()} -> {color}")


def set_friction(stage, static_friction, dynamic_friction):
    """PhysicsMaterial 생성 후 모든 Mesh에 바인딩"""
    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        children = list(stage.GetPseudoRoot().GetChildren())
        if not children:
            print("  오류: 루트 프림을 찾을 수 없습니다.")
            return
        root_prim = children[0]

    # PhysicsMaterial 프림 생성
    mat_path = root_prim.GetPath().AppendChild("PhysicsMaterial")
    mat_prim = stage.DefinePrim(mat_path, "Material")
    UsdPhysics.MaterialAPI.Apply(mat_prim)
    physics_mat = UsdPhysics.MaterialAPI(mat_prim)
    physics_mat.CreateStaticFrictionAttr().Set(static_friction)
    physics_mat.CreateDynamicFrictionAttr().Set(dynamic_friction)
    print(f"  PhysicsMaterial 생성: {mat_path}")
    print(f"    staticFriction: {static_friction}, dynamicFriction: {dynamic_friction}")

    # 모든 Mesh에 material 바인딩
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            mat = UsdShade.Material(mat_prim)
            binding.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            print(f"  마찰 바인딩: {prim.GetPath()}")


def process_file(filename, color, friction=None):
    filepath = os.path.join(ASSETS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"[{filename}] 오류: 파일을 찾을 수 없습니다.")
        return

    print(f"[{filename}]")
    stage = Usd.Stage.Open(filepath)

    set_display_color(stage, color)

    if friction is not None:
        set_friction(stage, static_friction=friction, dynamic_friction=friction)

    stage.GetRootLayer().Save()
    print(f"[{filename}] 저장 완료!\n")


if __name__ == "__main__":
    # cylinder, cylinder_case: 색상 (0.05,0.05,0.05), friction 0.5
    process_file("cylinder.usdc", color=(0.02, 0.02, 0.02), friction=0.5)
    process_file("cylinder_case.usdc", color=(0.02, 0.02, 0.02), friction=0.5)

    # Tseollo_integration: 색상 (0.1,0.1,0.1), friction 없음
    process_file("Tseollo_integration.usdc", color=(0.1, 0.1, 0.1))

    print("완료!")
