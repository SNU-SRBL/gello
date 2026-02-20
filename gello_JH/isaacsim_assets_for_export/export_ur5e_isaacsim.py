"""
Isaac Sim 내부에서 실행해야 하는 스크립트
Isaac Sim GUI의 Script Editor에서 실행하거나,
./isaaclab.sh -p 로 실행
"""

import omni.usd
from pxr import Usd, Sdf, UsdGeom
import omni.kit.commands

# 소스 USD 열기 (이미 열려있다면 생략)
src_path = "/home/Isaac/workspace/HD/Real2Sim/isaacsim_assets_for_export/Collected_World0/World0.usd"
dst_path = "/home/Isaac/workspace/HD/Real2Sim/isaacsim_assets_for_export/ur5e_tesollo_v2.usd"

# 현재 stage 가져오기
stage = omni.usd.get_context().get_stage()

# 만약 다른 파일이 열려있다면, 원하는 파일 열기
current_path = stage.GetRootLayer().identifier if stage else None
if current_path != src_path:
    print(f"Opening: {src_path}")
    omni.usd.get_context().open_stage(src_path)
    import asyncio
    asyncio.ensure_future(wait_and_export())
else:
    do_export()

async def wait_and_export():
    """Stage가 완전히 로드될 때까지 대기"""
    import omni.kit.app
    await omni.kit.app.get_app().next_update_async()
    await omni.kit.app.get_app().next_update_async()
    do_export()

def do_export():
    stage = omni.usd.get_context().get_stage()

    # ur5e prim 선택
    ur5e_path = "/World/simulation_env/ur5e"

    # Export with flatten using omni.kit.commands
    print(f"Exporting {ur5e_path} to {dst_path}")

    # Method 1: Using USD Composer's export
    try:
        import omni.kit.window.file_exporter as exporter
        # This might not work directly
    except:
        pass

    # Method 2: Using prim utils
    try:
        from omni.kit.commands import execute

        # 먼저 불필요한 prim 삭제
        paths_to_delete = [
            "/World/simulation_env/ur5e/Gripper",
            "/World/simulation_env/ur5e/Merged_Robot_flange",
        ]
        for path in paths_to_delete:
            if stage.GetPrimAtPath(path):
                execute("DeletePrims", paths=[path])
                print(f"Deleted: {path}")

        # Export selected prims
        # omni.kit.commands.execute("ExportPrim", ...)

    except Exception as e:
        print(f"Method 2 failed: {e}")

    # Method 3: Manual flatten with instance resolution
    try:
        from pxr import UsdUtils

        # Flatten하면서 instance를 해제
        # UsdUtils.FlattenLayerStack 사용
        flattened = UsdUtils.FlattenLayerStack(stage)

        # 새 stage에 저장
        flattened.Export(dst_path)
        print(f"Exported to: {dst_path}")

    except Exception as e:
        print(f"Method 3 failed: {e}")

    print("Export complete!")

# 직접 실행 시
if __name__ == "__main__":
    do_export()
