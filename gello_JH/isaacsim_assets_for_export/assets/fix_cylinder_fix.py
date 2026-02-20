"""cylinder_fix.usdc: scale 1/1000, 색상 (0.02,0.02,0.02), friction 0.5, collision 추가"""
import os
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Gf, Vt, Sdf

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cylinder_fix.usdc")
stage = Usd.Stage.Open(filepath)
root = stage.GetDefaultPrim() or list(stage.GetPseudoRoot().GetChildren())[0]

# 1. Scale 1/1000
xformable = UsdGeom.Xformable(root)
for op in xformable.GetOrderedXformOps():
    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
        op.Set(Gf.Vec3d(0.001, 0.001, 0.001))
        print("Scale (0.001, 0.001, 0.001) 설정 완료 (기존 op 수정)")
        break
else:
    scale_op = xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
    scale_op.Set(Gf.Vec3d(0.001, 0.001, 0.001))
    print("Scale (0.001, 0.001, 0.001) 설정 완료 (새 op 추가)")

# 2. PhysicsMaterial (friction 0.5)
mat_path = root.GetPath().AppendChild("PhysicsMaterial")
mat_prim = stage.DefinePrim(mat_path, "Material")
UsdPhysics.MaterialAPI.Apply(mat_prim)
physics_mat = UsdPhysics.MaterialAPI(mat_prim)
physics_mat.CreateStaticFrictionAttr().Set(0.5)
physics_mat.CreateDynamicFrictionAttr().Set(0.5)
print("PhysicsMaterial (friction 0.5) 생성 완료")

# 3. Mesh에 색상, collision, material 바인딩
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        # 색상
        UsdGeom.Mesh(prim).GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.02, 0.02, 0.02)]))
        # Collision
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr().Set("convexDecomposition")
        # PhysxConvexDecompositionCollisionAPI 적용
        api_schemas = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
        prepended = list(api_schemas.prependedItems)
        if "PhysxConvexDecompositionCollisionAPI" not in prepended:
            prepended.append("PhysxConvexDecompositionCollisionAPI")
            api_schemas.prependedItems = prepended
            prim.SetMetadata("apiSchemas", api_schemas)
        prim.CreateAttribute("physxConvexDecompositionCollision:maxConvexHulls", Sdf.ValueTypeNames.Int).Set(2)
        # Friction 바인딩
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(UsdShade.Material(mat_prim), UsdShade.Tokens.weakerThanDescendants, "physics")
        print(f"Mesh 설정 완료: {prim.GetPath()}")

stage.GetRootLayer().Save()
print("\n저장 완료!")
