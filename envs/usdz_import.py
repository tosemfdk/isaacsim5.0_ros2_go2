from pxr import UsdPhysics, PhysxSchema, UsdGeom, Gf
import omni.usd
import os

USDZ_PATH = "/home/loe/workspace/github/isaacsim5.0_ros2_go2/models/konkuk_library.usdz"

def GS_import():
    if os.path.exists(USDZ_PATH):
        omni.usd.get_context().open_stage(USDZ_PATH)
        stage = omni.usd.get_context().get_stage()
        print(f"[INFO]: Loaded NuRec scene from {USDZ_PATH}")
        

        gauss_prim = stage.GetPrimAtPath("/World/gauss")
        
        if gauss_prim.IsValid():
            xformable = UsdGeom.Xformable(gauss_prim)
            
            # 위치 (Translate): 0.0, 0.0, 50.0
            if not gauss_prim.GetAttribute("xformOp:translate"):
                xformable.AddTranslateOp()
            gauss_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3f(0.0, 0.0, 1.4))
            


            if not gauss_prim.GetAttribute("xformOp:rotateXYZ"):
                xformable.AddRotateXYZOp()
            gauss_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(17.3, 0.0, 0.0))
            
            # 크기 (Scale):
            if not gauss_prim.GetAttribute("xformOp:scale"):
                xformable.AddScaleOp()
            gauss_prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(3.0, 3.0, 3.0))
            
            # 인스턴스 가능 여부 설정 (Instanceable: False)
            gauss_prim.SetInstanceable(False)
            
            print(f"[INFO]: Applied transforms to {gauss_prim.GetPath()}")
        else:
            print("[ERROR]: Prim at '/World/gauss' not found. Please check the path inside USDZ.")

        # 4. 물리 장면을 'Synchronous'로 설정 (내비게이션 안정성 확보) [3, 4]
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
                physx_scene.GetUpdateTypeAttr().Set("Synchronous")
                break
    else:
        print(f"[ERROR]: USDZ file not found at {USDZ_PATH}")