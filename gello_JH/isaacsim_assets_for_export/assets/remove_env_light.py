"""env_light 프림 제거 스크립트"""
import os
from pxr import Usd

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_FILES = ["cylinder.usdc", "cylinder_case.usdc", "Tseollo_integration.usdc"]

for filename in TARGET_FILES:
    filepath = os.path.join(ASSETS_DIR, filename)
    stage = Usd.Stage.Open(filepath)
    root = stage.GetDefaultPrim() or list(stage.GetPseudoRoot().GetChildren())[0]
    light_path = root.GetPath().AppendChild("env_light")
    prim = stage.GetPrimAtPath(light_path)
    if prim.IsValid():
        stage.RemovePrim(light_path)
        stage.GetRootLayer().Save()
        print(f"[{filename}] env_light 제거 완료")
    else:
        print(f"[{filename}] env_light 없음")

print("\n완료!")
