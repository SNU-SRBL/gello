"""
USD 파일 1/1000 스케일링 스크립트
- 대상: cylinder.usdc, cylinder_case.usdc, Tseollo_integration.usdc
- 루트 프림에 scale (0.001, 0.001, 0.001) 적용
- 원본은 *_backup.usdc로 백업
"""
import os
import shutil
from pxr import Usd, UsdGeom, Gf

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
SCALE_FACTOR = 0.001

TARGET_FILES = [
    "cylinder.usdc",
    "cylinder_case.usdc",
    "Tseollo_integration.usdc",
]


def scale_usd(file_path):
    basename = os.path.basename(file_path)
    name, ext = os.path.splitext(file_path)
    backup_path = f"{name}_backup{ext}"

    # 백업 생성
    shutil.copy2(file_path, backup_path)
    print(f"[{basename}] 백업 생성: {os.path.basename(backup_path)}")

    stage = Usd.Stage.Open(file_path)
    root_prim = stage.GetDefaultPrim()
    if not root_prim or not root_prim.IsValid():
        # DefaultPrim이 없으면 첫 번째 자식 프림 사용
        children = list(stage.GetPseudoRoot().GetChildren())
        if not children:
            print(f"[{basename}] 오류: 프림을 찾을 수 없습니다.")
            return False
        root_prim = children[0]
        print(f"[{basename}] DefaultPrim 없음, 사용: {root_prim.GetPath()}")

    xformable = UsdGeom.Xformable(root_prim)
    existing_ops = xformable.GetOrderedXformOps()

    # 기존 Scale op 찾기
    scale_found = False
    for op in existing_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            old_scale = op.Get()
            new_scale = Gf.Vec3d(
                old_scale[0] * SCALE_FACTOR,
                old_scale[1] * SCALE_FACTOR,
                old_scale[2] * SCALE_FACTOR,
            )
            op.Set(new_scale)
            print(f"[{basename}] 스케일 변경: {old_scale} -> {new_scale}")
            scale_found = True
            break

    # Scale op이 없으면 새로 추가
    if not scale_found:
        scale_op = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
        scale_op.Set(Gf.Vec3d(SCALE_FACTOR, SCALE_FACTOR, SCALE_FACTOR))
        print(f"[{basename}] 새 스케일 추가: ({SCALE_FACTOR}, {SCALE_FACTOR}, {SCALE_FACTOR})")

    stage.GetRootLayer().Save()
    print(f"[{basename}] 저장 완료!")
    return True


if __name__ == "__main__":
    print(f"스케일 팩터: {SCALE_FACTOR} (1/{int(1/SCALE_FACTOR)})")
    print(f"대상 디렉토리: {ASSETS_DIR}\n")

    for filename in TARGET_FILES:
        filepath = os.path.join(ASSETS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[{filename}] 오류: 파일을 찾을 수 없습니다.")
            continue
        scale_usd(filepath)
        print()

    print("완료!")
