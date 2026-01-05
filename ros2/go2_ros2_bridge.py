import omni.graph.core as og
import omni.replicator.core as rep
import omni.kit.commands

class RobotDataManager:
    def __init__(self, env, lidar_sensors, cameras, cfg):
        self.env = env
        self.num_envs = cfg.num_envs
        self.lidar_sensors = lidar_sensors
        self.cameras = cameras
        
        # 1. 카메라 발행기 설정 (루프 처리)
        self._setup_camera_publishers()
        
        # 2. 라이다 발행기 설정 (루프 처리)
        self._setup_lidar_publishers()
        
        # 3. 공통 데이터(Clock 등) 발행용 OmniGraph 설정
        self._setup_global_omnigraph()

        # self._setup_odom_publishers()

        # self._setup_tf_and_joint_publishers()

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
                render_product_obj = rep.create.render_product(annotator.GetPath(), resolution=[1024, 1024], name="Isaac")
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
                            ("LidarHelper.inputs:resetSimulationTimeOnStop", True),
                        ],
                    },
                )
                print(f"[Bridge] LiDAR Helper Node (Env {i}) created successfully.")


    def _setup_odom_publishers(self):
            """로봇의 상태를 계산하여 ROS 2 Odometry 토픽으로 발행"""
            for i in range(self.num_envs):
                # 로봇의 베이스 프라이드 경로 (이전 코드 기준)
                robot_base_path = f"/World/envs/env_{i}/Go2/base"
                graph_path = f"/World/Odom_ROS_Graph_env_{i}"
                
                try:
                    og.Controller.edit(
                        {"graph_path": graph_path, "evaluator_name": "execution"},
                        {
                            og.Controller.Keys.CREATE_NODES: [
                                ("OnTick", "omni.graph.action.OnTick"),
                                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                                ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
                                ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                            ],
                            og.Controller.Keys.SET_VALUES: [
                                ("ComputeOdom.inputs:chassisPrim", robot_base_path),
                                ("PublishOdom.inputs:topicName", f"env_{i}/unitree_go2/odom"),
                                ("PublishOdom.inputs:odomFrameId", f"odom_{i}"),
                                ("PublishOdom.inputs:chassisFrameId", f"base_link_{i}"),
                            ],
                            og.Controller.Keys.CONNECT: [
                                ("OnTick.outputs:tick", "ComputeOdom.inputs:execIn"),
                                ("ComputeOdom.outputs:execOut", "PublishOdom.inputs:execIn"),
                                ("ComputeOdom.outputs:position", "PublishOdom.inputs:position"),
                                ("ComputeOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
                                ("ComputeOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
                                ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
                                ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
                            ],
                        },
                    )
                    print(f"[Bridge] Odometry Publisher created for Env {i}")
                except Exception as e:
                    print(f"[Error] Failed to create Odometry Helper for Env {i}: {e}")



    def _setup_tf_and_joint_publishers(self):
            """TF Tree 및 Joint States 발행 설정"""
            for i in range(self.num_envs):
                physics_base_path = f"/World/envs/env_{i}/Go2/base" # 로봇 루트 (Articulation)
                graph_path = f"/World/TF_Joint_Graph_env_{i}"
                
                try:
                    og.Controller.edit(
                        {"graph_path": graph_path, "evaluator_name": "execution"},
                        {
                            og.Controller.Keys.CREATE_NODES: [
                                ("OnTick", "omni.graph.action.OnTick"),
                                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                                # 1. TF 발행 노드 (Odom -> Base_link 등 동적 변환)
                                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                                # 2. Static TF 발행 노드 (고정된 관절 구조)
                                ("PublishStaticTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                                # 3. Joint State 발행 노드
                                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                            ],
                            og.Controller.Keys.SET_VALUES: [
                                # [TF 설정] 동적: odom_{i} -> base_link_{i}
                                ("PublishTF.inputs:parentPrim", f"/World/envs/env_{i}"), # odom 대신 환경 루트를 부모로 설정 가능
                                ("PublishTF.inputs:targetPrims", [physics_base_path]),
                                ("PublishTF.inputs:nodeNamespace", f"env_{i}"),
                                
                                # [Static TF 설정] 고정: 로봇의 전체 관절 트리
                                ("PublishStaticTF.inputs:targetPrims", [physics_base_path]),
                                ("PublishStaticTF.inputs:staticPublisher", True), # Static Broadcaster 역할
                                ("PublishStaticTF.inputs:nodeNamespace", f"env_{i}"),
                                
                                # [Joint State 설정]
                                ("PublishJointState.inputs:targetPrim", physics_base_path),
                                ("PublishJointState.inputs:topicName", f"env_{i}/joint_states"),
                            ],
                            og.Controller.Keys.CONNECT: [
                                ("OnTick.outputs:tick", "PublishTF.inputs:execIn"),
                                ("OnTick.outputs:tick", "PublishStaticTF.inputs:execIn"),
                                ("OnTick.outputs:tick", "PublishJointState.inputs:execIn"),
                                
                                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                                ("ReadSimTime.outputs:simulationTime", "PublishStaticTF.inputs:timeStamp"),
                                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                            ],
                        },
                    )
                    print(f"[Bridge] TF and JointState Publishers created for Env {i}")
                except Exception as e:
                    print(f"[Error] Failed to create TF/Joint Helpers for Env {i}: {e}")




    def _setup_global_omnigraph(self):
        """시뮬레이션 시간(/clock) 발행 (모든 환경 공통)"""
        try:
            og.Controller.edit(
                {"graph_path": "/World/Push_ROS2_Clock", "evaluator_name": "execution"},
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

