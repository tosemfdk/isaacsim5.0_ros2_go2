import omni.graph.core as og
import omni.replicator.core as rep
import omni.kit.commands
import rclpy
from geometry_msgs.msg import Twist
import go2.go2_ctrl as go2_ctrl

class RobotDataManager:
    def __init__(self, env, lidar_sensors, cameras, cfg):
        self.env = env
        self.num_envs = cfg.num_envs
        self.lidar_sensors = lidar_sensors
        self.cameras = cameras
        
        # ROS 2 노드 초기화 (구독자용)
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("go2_robot_data_manager")

        # 1. 카메라 발행기 설정 (루프 처리)
        self._setup_camera_publishers()
        
        # 2. 라이다 발행기 설정 (루프 처리)
        self._setup_lidar_publishers()
        
        # 3. 공통 데이터(Clock 등) 발행용 OmniGraph 설정
        self._setup_global_omnigraph()

        self._setup_odom_publishers()

        # 4. 속도 명령(cmd_vel) 구독자 설정 (다중 환경인 경우)
        if self.num_envs > 1:
            self._setup_cmd_vel_subscribers()

    def _setup_cmd_vel_subscribers(self):
        """각 환경별 cmd_vel 토픽 구독 설정"""
        self._velocity_subs = []
        for i in range(self.num_envs):
            topic_name = f"env_{i}/unitree_go2/cmd_vel"
            # 클로저를 사용하여 인덱스 i를 캡처
            sub = self.node.create_subscription(
                Twist,
                topic_name,
                lambda msg, idx=i: self._cmd_vel_callback(msg, idx),
                10
            )
            self._velocity_subs.append(sub)
            print(f"[Bridge] Subscribed to velocity commands on: {topic_name}")

    def _cmd_vel_callback(self, msg, idx):
        """수신된 Twist 메시지를 go2_ctrl의 전역 텐서에 주입"""
        if go2_ctrl.base_vel_cmd_input is not None:
            go2_ctrl.base_vel_cmd_input[idx, 0] = msg.linear.x
            go2_ctrl.base_vel_cmd_input[idx, 1] = msg.linear.y
            go2_ctrl.base_vel_cmd_input[idx, 2] = msg.angular.z

    def _setup_camera_publishers(self):
        """라이터 대신 OmniGraph의 ROS2 Camera Helper 노드를 직접 생성"""
        if self.cameras is not None:
            for i in range(self.num_envs):
                # Render Product 경로 가져오기
                render_product_path = self.cameras.render_product_paths[i]
                
                # 각 환경별 고유한 그래프 경로
                graph_path = f"/World/Camera_ROS_Graph_env_{i}"
                
                try:
                    # OmniGraph 구성
                    og.Controller.edit(
                        {"graph_path": graph_path, "evaluator_name": "push"},
                        {
                            og.Controller.Keys.CREATE_NODES: [
                                ("OnTick", "omni.graph.action.OnTick"),
                                ("cameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                            ],
                            og.Controller.Keys.CONNECT: [
                                ("OnTick.outputs:tick", "cameraHelperRgb.inputs:execIn"),
                            ],
                            og.Controller.Keys.SET_VALUES: [
                                ("cameraHelperRgb.inputs:renderProductPath", render_product_path),
                                ("cameraHelperRgb.inputs:frameId", f"front_cam_link_{i}"),
                                ("cameraHelperRgb.inputs:topicName", f"env_{i}/unitree_go2/front_cam/rgb"),
                                ("cameraHelperRgb.inputs:type", "rgb"),
                                ("cameraHelperRgb.inputs:frameSkipCount", 5),
                            ],
                        },
                    )
                    print(f"[Bridge] RGB Publisher created for Env {i} via OmniGraph")
                except Exception as e:
                    print(f"[Error] Failed to create Camera Helper for Env {i}: {e}")

    def _setup_lidar_publishers(self):
        if self.lidar_sensors is not None:
            for i, annotator in enumerate(self.lidar_sensors):
                # 1. 렌더 제품 경로 가져오기 (문서의 renderProductPath 입력값)
                render_product_obj = rep.create.render_product(annotator.GetPath(), resolution=[1024, 64], name="Isaac")
                render_product_path = render_product_obj.path
                # 2. 각 환경별 고유 그래프 경로
                graph_path = f"/World/Lidar_ROS_Graph_env_{i}"
                
                # 3. OmniGraph 노드 생성 및 설정
                og.Controller.edit(
                    {"graph_path": graph_path, "evaluator_name": "push"},
                    {
                        og.Controller.Keys.CREATE_NODES: [
                            ("OnTick", "omni.graph.action.OnTick"),
                            ("LidarHelper", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
                        ],
                        og.Controller.Keys.CONNECT: [
                            ("OnTick.outputs:tick", "LidarHelper.inputs:execIn"),
                        ],
                        og.Controller.Keys.SET_VALUES: [
                            ("LidarHelper.inputs:renderProductPath", render_product_path),
                            ("LidarHelper.inputs:topicName", f"env_{i}/unitree_go2/lidar/point_cloud"),
                            ("LidarHelper.inputs:frameId", f"go2_lidar{i}"),
                            ("LidarHelper.inputs:type", "point_cloud"), 
                            ("LidarHelper.inputs:fullScan", True), 
                            ("LidarHelper.inputs:frameSkipCount", 10),
                            ("LidarHelper.inputs:resetSimulationTimeOnStop", True),
                        ],
                    },
                )
                print(f"[Bridge] LiDAR Helper Node (Env {i}) created successfully.")




    def _setup_odom_publishers(self):
        """데이터 주입을 위한 ROS 2 Publish Odometry 노드만 생성"""
        for i in range(self.num_envs):
            graph_path = f"/World/Odom_ROS_Graph_env_{i}"
            
            try:
                og.Controller.edit(
                    {"graph_path": graph_path, "evaluator_name": "push"},
                    {
                        og.Controller.Keys.CREATE_NODES: [
                            # OnTick은 이제 필요하지 않습니다. Python에서 직접 실행(execIn)을 때려줄 것이기 때문입니다.
                            ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                        ],
                        og.Controller.Keys.SET_VALUES: [
                            ("PublishOdom.inputs:topicName", f"env_{i}/unitree_go2/odom"),
                            ("PublishOdom.inputs:odomFrameId", f"odom_{i}"),
                            ("PublishOdom.inputs:chassisFrameId", f"base_link_{i}"),
                        ],
                    },
                )
                print(f"[Bridge] Odometry Publisher Node created for Env {i}")
            except Exception as e:
                print(f"[Error] Failed to create Odometry Publisher for Env {i}: {e}")




    def _setup_global_omnigraph(self):
        """시뮬레이션 시간(/clock) 발행 (모든 환경 공통)"""
        try:
            og.Controller.edit(
                {"graph_path": "/World/Push_ROS2_Clock", "evaluator_name": "execpushution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ],
                },
            )
        except Exception as e:
            print(f"[Error] Global OmniGraph setup error: {e}")

    # RobotDataManager 클래스 내부
    def update(self):
        """매 프레임마다 호출되어 ROS 2 데이터를 업데이트"""
        # ROS 2 콜백 처리 (비동기 메시지 수신)
        rclpy.spin_once(self.node, timeout_sec=0)

        # Isaac Lab Tensor 데이터 가져오기 (unitree_go2는 Scene에 등록된 이름)
        robot_data = self.env.unwrapped.scene["go2"].data
        
        # 시뮬레이션 시간 가져오기
        current_sim_time = self.env.unwrapped.sim.current_time

        for i in range(self.num_envs):
            # 그래프 상의 PublishOdom 노드 경로
            odom_node_path = f"/World/Odom_ROS_Graph_env_{i}/PublishOdom"
            
            # 1. 위치 및 자세 (Root State: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z])
            pos = robot_data.root_state_w[i, :3].tolist()
            # Isaac Lab(WXYZ) -> ROS 2(XYZW) 변환
            quat_wxyz = robot_data.root_state_w[i, 3:7]
            quat_xyzw = [quat_wxyz[1].item(), quat_wxyz[2].item(), quat_wxyz[3].item(), quat_wxyz[0].item()]
            
            # 2. 선속도 및 각속도 (Body Frame 기준)
            lin_vel = robot_data.root_lin_vel_b[i].tolist()
            ang_vel = robot_data.root_ang_vel_b[i].tolist()

            try:
                # OmniGraph 노드 속성에 직접 값 주입 (GPU -> ROS Bridge 데이터 흐름)
                og.Controller.set(og.Controller.attribute(f"{odom_node_path}.inputs:position"), pos)
                og.Controller.set(og.Controller.attribute(f"{odom_node_path}.inputs:orientation"), quat_xyzw)
                og.Controller.set(og.Controller.attribute(f"{odom_node_path}.inputs:linearVelocity"), lin_vel)
                og.Controller.set(og.Controller.attribute(f"{odom_node_path}.inputs:angularVelocity"), ang_vel)
                og.Controller.set(og.Controller.attribute(f"{odom_node_path}.inputs:timeStamp"), current_sim_time)
                
                # TF 업데이트가 필요한 경우 여기서 추가로 처리 가능
            except Exception as e:
                # 시뮬레이션 초기화 단계에서 노드가 아직 없을 때 에러 방지
                pass

    def destroy_node(self):
        """시뮬레이션 종료 시 노드 정리"""
        if hasattr(self, 'node'):
            self.node.destroy_node()