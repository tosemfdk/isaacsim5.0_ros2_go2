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
from envs.usdz_import import GS_import

FILE_PATH = os.path.join(os.path.dirname(__file__), "config")
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):
    go2_env_cfg = Go2RLEnvCfg()
    go2_env_cfg.decimation = math.ceil(1./go2_env_cfg.sim.dt/cfg.freq)
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation

    env, policy = go2_rl_env(go2_env_cfg, cfg)
    
    #run simulation
    dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    obs, _ = env.reset()


    print("[INFO]: simulation started")


    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

        elapsed_time = time.time() - start_time

        sleep_time = dt - (elapsed_time)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, dt/elapsed_time)
        print(f"\rStep time: {actual_loop_time*1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)
    simulation_app.close()

if __name__ == "__main__":
    GS_import()
    run_simulator()
    