"""Mesh에 CollisionAPI 추가 스크립트"""
import os
from pxr import Usd, UsdGeom, UsdPhysics

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_FILES = ["cylinder.usdc", "cylinder_case.usdc", "Tseollo_integration.usdc"]

for filename in TARGET_FILES:
    filepath = os.path.join(ASSETS_DIR, filename)
    stage = Usd.Stage.Open(filepath)
    print(f"[{filename}]")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_col = UsdPhysics.MeshCollisionAPI(prim)
            mesh_col.CreateApproximationAttr().Set("convexHull")
            print(f"  CollisionAPI 추가: {prim.GetPath()} (convexHull)")
    stage.GetRootLayer().Save()
    print(f"  저장 완료\n")

print("완료!")
