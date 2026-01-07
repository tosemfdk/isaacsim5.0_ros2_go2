import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import hydra
from omegaconf import DictConfig
import tkinter as tk
from tkinter import ttk
import threading

# 설정 파일 경로 설정 (sim.yaml이 있는 위치)
FILE_PATH = os.path.join(os.path.dirname(__file__), "config")

class Go2MultiCmdVelPublisher(Node):
    def __init__(self, cfg):
        super().__init__('go2_multi_cmd_vel_publisher')
        self.num_envs = cfg.num_envs
        self.get_logger().info(f'Initializing CmdVel GUI for {self.num_envs} environments...')

        self.publishers_list = []
        self.vel_commands = [] # 각 환경별 속도 저장용 리스트 (dict 형태)

        for i in range(self.num_envs):
            # go2_ros2_bridge.py에서 구독하는 토픽명과 일치시킴
            topic_name = f'/env_{i}/unitree_go2/cmd_vel'
            pub = self.create_publisher(Twist, topic_name, 10)
            self.publishers_list.append(pub)
            self.vel_commands.append({'x': 0.0, 'y': 0.0, 'z': 0.0})
            self.get_logger().info(f'Created publisher for: {topic_name}')

        # 20Hz(0.05초) 주기로 현재 설정된 모든 로봇의 속도 발행
        self.timer = self.create_timer(0.05, self.publish_all_vels)

    def publish_all_vels(self):
        for i in range(self.num_envs):
            msg = Twist()
            msg.linear.x = float(self.vel_commands[i]['x'])
            msg.linear.y = float(self.vel_commands[i]['y'])
            msg.angular.z = float(self.vel_commands[i]['z'])
            self.publishers_list[i].publish(msg)

    def update_velocity(self, env_idx, axis, val):
        """GUI 슬라이더에서 호출되는 콜백"""
        self.vel_commands[env_idx][axis] = float(val)

def start_gui(node, num_envs):
    root = tk.Tk()
    root.title("Isaac Lab Go2 Multi-Robot Teleop")
    root.geometry("450x700")

    # 스크롤 가능한 영역 설정 (로봇이 많을 경우 대비)
    main_frame = ttk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=1)

    canvas = tk.Canvas(main_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    scrollable_frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    for i in range(num_envs):
        group = ttk.LabelFrame(scrollable_frame, text=f"Robot Environment {i}")
        group.pack(padx=15, pady=10, fill="x")

        # Linear X Slider
        ttk.Label(group, text="Linear X (Forward/Backward)").pack()
        s_x = ttk.Scale(group, from_=-1.5, to=1.5, orient=tk.HORIZONTAL, 
                        command=lambda v, idx=i: node.update_velocity(idx, 'x', v))
        s_x.set(0.0)
        s_x.pack(fill="x", padx=10)

        # Linear Y Slider
        ttk.Label(group, text="Linear Y (Lateral)").pack()
        s_y = ttk.Scale(group, from_=-1.0, to=1.0, orient=tk.HORIZONTAL, 
                        command=lambda v, idx=i: node.update_velocity(idx, 'y', v))
        s_y.set(0.0)
        s_y.pack(fill="x", padx=10)

        # Angular Z Slider
        ttk.Label(group, text="Angular Z (Yaw)").pack()
        s_z = ttk.Scale(group, from_=-2.0, to=2.0, orient=tk.HORIZONTAL, 
                        command=lambda v, idx=i: node.update_velocity(idx, 'z', v))
        s_z.set(0.0)
        s_z.pack(fill="x", padx=10)

        # 개별 정지 버튼
        def reset_robot(idx, sx, sy, sz):
            sx.set(0.0)
            sy.set(0.0)
            sz.set(0.0)
            node.update_velocity(idx, 'x', 0.0)
            node.update_velocity(idx, 'y', 0.0)
            node.update_velocity(idx, 'z', 0.0)

        ttk.Button(group, text=f"STOP Robot {i}", 
                   command=lambda idx=i, sx=s_x, sy=s_y, sz=s_z: reset_robot(idx, sx, sy, sz)).pack(pady=5)

    root.mainloop()

@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def main(cfg: DictConfig):
    rclpy.init()
    node = Go2MultiCmdVelPublisher(cfg)
    
    # ROS 2 스핀을 별도 스레드에서 실행하여 GUI 루프와 병행 처리
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    
    try:
        start_gui(node, cfg.num_envs)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()