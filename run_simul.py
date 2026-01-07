import os
import hydra
import torch
import time
import math
import argparse
from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Unitree go2 ros2 setup")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.kit_args = "--/renderer/multiGpu/enabled=true --/renderer/multiGpu/maxGpuCount=2"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()

# 확장 프로그램 활성화 (최신 ID 사용 권장)
extension_id = "isaacsim.ros2.bridge" 
ext_manager.set_extension_enabled_immediate(extension_id, True)


"""Rest everything follows."""

import torch
import time
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext


import omni
import carb
import go2.go2_ctrl as go2_ctrl
from go2.go2_env import go2_rl_env, Go2RLEnvCfg
from envs.usdz_import import GS_import
from go2.go2_sensors import sensor_manager
import ros2.go2_ros2_bridge as go2_ros2_bridge
import envs.sim_env as sim_env


FILE_PATH = os.path.join(os.path.dirname(__file__), "config")
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    go2_env_cfg = Go2RLEnvCfg()
    go2_env_cfg.decimation = math.ceil(1./go2_env_cfg.sim.dt/cfg.freq)
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
    go2_env_cfg.scene.num_envs = cfg.num_envs
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    
    # Apply sensor transforms from config to environment
    go2_env_cfg.scene.front_cam.offset.pos = tuple(cfg.sensor.camera.pos)
    go2_env_cfg.scene.front_cam.offset.rot = tuple(cfg.sensor.camera.rot)

    env, policy = go2_rl_env(go2_env_cfg, cfg)


    # Simulation environment
    if (cfg.env_name == "warehouse"):
        sim_env.create_warehouse_env() # warehouse
    elif (cfg.env_name == "warehouse-forklifts"):
        sim_env.create_warehouse_forklifts_env() # warehouse forklifts
    elif (cfg.env_name == "warehouse-shelves"):
        sim_env.create_warehouse_shelves_env() # warehouse shelves
    elif (cfg.env_name == "full-warehouse"):
        sim_env.create_full_warehouse_env() # full warehouse
    elif (cfg.env_name == "office"):
        sim_env.create_office_env() # office

    # Sensor setup
    sm = sensor_manager(cfg)
    # cameras, lidars
    lidars = sm.create_lidar()
    cameras = env.unwrapped.scene["front_cam"]

    # ROS2 Bridge
    dm = go2_ros2_bridge.RobotDataManager(env, lidars, cameras, cfg)

    # Keyboard control
    system_input = carb.input.acquire_input_interface()
    system_input.subscribe_to_keyboard_events(omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event)
    
    print("[INFO]: simulation started")

    obs, _ = env.reset()

    #run simulation
    dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)



    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
                
                actions = policy(obs)

                obs, _, _, _ = env.step(actions)


        dm.update()
        # dm.update_tf()
        elapsed_time = time.time() - start_time

        sleep_time = dt - (elapsed_time)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, dt/elapsed_time)
        print(f"\rStep time: {actual_loop_time*1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)
    dm.destroy_node()
    simulation_app.close()

if __name__ == "__main__":
    # GS_import()
    run_simulator()
    