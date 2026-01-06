from isaacsim.core.utils.prims import define_prim, get_prim_at_path
try:
    import isaacsim.storage.native as nucleus_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.nucleus as nucleus_utils

import omni.replicator.core as rep
from isaaclab.sim.schemas import CollisionPropertiesCfg, define_collision_properties
from isaaclab.sim.utils import apply_nested
import isaaclab.sim as sim_utils
import omni
from pxr import UsdGeom, Usd

def add_semantic_label():
    # ground_plane = rep.get.prims("/World/ground")
    # with ground_plane:
    # # Add a semantic label
    #     rep.modify.semantics([("class", "floor")])
    pass

def add_collision(prim_path):

    # 1. 최소한의 속성만 정의 (None은 수정되지 않음)
    collision_cfg = CollisionPropertiesCfg(
        collision_enabled=True,   
        contact_offset=0.02,
        rest_offset=0.0,
        torsional_patch_radius=None, # 불필요한 연산 방지
        min_torsional_patch_radius=None
    )

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)

    # 2. USD PrimRange를 사용하여 효율적으로 하위 Mesh 탐색
    for prim in Usd.PrimRange(root_prim):
        # Mesh 타입이고, 프로토타입(인스턴스 원본)이 아닌 경우에만 스키마 적용
        if prim.IsA(UsdGeom.Mesh) and not prim.IsPrototype():
            try:
                define_collision_properties(prim.GetPath().pathString, collision_cfg)
            except Exception:
                continue

def create_warehouse_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim_path = "/World/Warehouse"
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    prim.GetReferences().AddReference(asset_path)
    add_collision(prim_path)

def create_warehouse_forklifts_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd"
    prim.GetReferences().AddReference(asset_path)

def create_warehouse_shelves_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
    prim.GetReferences().AddReference(asset_path)

def create_full_warehouse_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Warehouse")
    prim = define_prim("/World/Warehouse", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
    prim.GetReferences().AddReference(asset_path)

def create_hospital_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Hospital")
    prim = define_prim("/World/Hospital", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Hospital/hospital.usd"
    prim.GetReferences().AddReference(asset_path)

def create_office_env():
    add_semantic_label()
    assets_root_path = nucleus_utils.get_assets_root_path()
    prim = get_prim_at_path("/World/Office")
    prim = define_prim("/World/Office", "Xform")
    asset_path = assets_root_path+"/Isaac/Environments/Office/office.usd"
    prim.GetReferences().AddReference(asset_path)
    #CollisionPropertiesCfg를 통한 충돌 설정
    collision_cfg = CollisionPropertiesCfg(
        collision_enabled=True,   
        contact_offset=0.05,
        rest_offset=0.02           
    )

    #하위 모든 메시에 충돌 속성 전파
    apply_nested(lambda p: define_collision_properties(p, collision_cfg))(prim.GetPath())

    