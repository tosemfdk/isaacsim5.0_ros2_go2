import os
import torch
import carb
from isaaclab.envs import ManagerBasedEnv
from rsl_rl.runners import OnPolicyRunner
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg


base_vel_cmd_input = None
keyboard = None

# Initialize base_vel_cmd_input as a tensor when created
def init_base_vel_cmd(num_envs):
    global base_vel_cmd_input, keyboard
    base_vel_cmd_input = torch.zeros((num_envs, 3), dtype=torch.float32)
    # 환경이 하나일 때만 키보드 컨트롤러 활성화
    if num_envs == 1:
        keyboardCfg = Se2KeyboardCfg()
        keyboard = Se2Keyboard(keyboardCfg)

# Modify base_vel_cmd to use the tensor directly
def base_vel_cmd(env: ManagerBasedEnv) -> torch.Tensor:
    global base_vel_cmd_input, keyboard
    # 키보드가 활성화된 경우 (num_envs == 1), 텐서의 첫 번째 행을 키보드 입력으로 업데이트
    if keyboard is not None:
        vels = keyboard.advance() # [lin_x, lin_y, ang_z]
        base_vel_cmd_input[0, 0] = vels[0]
        base_vel_cmd_input[0, 1] = vels[1]
        base_vel_cmd_input[0, 2] = vels[2]
        
    return base_vel_cmd_input.clone().to(env.device)

# Update sub_keyboard_event to modify specific rows of the tensor based on key inputs
def sub_keyboard_event(event) -> bool:
    # Se2Keyboard 또는 ROS 2 cmd_vel이 제어권을 가지므로 기존 수동 이벤트 핸들러는 비워둡니다.
    return True