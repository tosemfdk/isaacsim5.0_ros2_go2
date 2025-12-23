import os
import hydra
import rclpy
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

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import time
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext

from go2.go2_env import go2_rl_env, Go2RLEnvCfg



    




FILE_PATH = os.path.join(os.path.dirname(__file__), "config")
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):

    env, policy = go2_rl_env(Go2RLEnvCfg(), cfg)
    obs, _ = env.get_observations()
    sim = env.unwrapped.sim 
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    print("[INFO]: simulation started")
    
    dt = env.unwrapped.step_dt

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

        sleep_time = dt - (time.time() - start_time)
        
        if sleep_time > 0:
            time.sleep(sleep_time)

    simulation_app.close()

if __name__ == "__main__":
    run_simulator()
    