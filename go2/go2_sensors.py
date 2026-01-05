from isaaclab.sensors import Camera, CameraCfg
import omni
import numpy as np
import isaaclab.sim as sim_utils
from pxr import Gf
import omni.replicator.core as rep


class sensor_manager:

    def __init__(self, num_envs):
        self.num_envs = num_envs

    def create_camera(self):
        cameras= []
        for env_idx in range(self.num_envs):

            # 1.(CameraCfg)
            camera_cfg = CameraCfg(
                prim_path=f"/World/envs/env_{env_idx}/Go2/base/front_cam",
                update_period=0.0,                      
                data_types=["rgb", "depth"],              
                spawn=sim_utils.PinholeCameraCfg(),             
                width=640,                                       
                height=480,                                      
                offset=CameraCfg.OffsetCfg(
                    pos=(0.5, 0.0, 0.1),                         
                    rot=(1.0, 0.0, 0.0, 0.0),                    
                    convention="world"                             
                ),
            )

            camera = Camera(cfg=camera_cfg)

            cameras.append(camera)

        return cameras



    def create_lidar(self):
        lidar_sensors = []

        sensor_attributes = {'omni:sensor:Core:scanRateBaseHz': 20}

        for env_idx in range(self.num_envs):

            # Create the RTX Lidar with the specified attributes.
            _, sensor = omni.kit.commands.execute(
                "IsaacSensorCreateRtxLidar",
                translation=Gf.Vec3d(0.0, 0.0, 0.25),
                orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
                path="lidar",
                parent=f"/World/envs/env_{env_idx}/Go2/base",
                config="Hesai_XT32_SD10",
                **sensor_attributes,
            )
        

            lidar_sensors.append(sensor)

        return lidar_sensors