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
        """공식 문서(ROS2RtxLidarHelper) 사양에 맞춘 라이다 설정"""
        if self.lidar_sensors is not None:
            for i, annotator in enumerate(self.lidar_sensors):
                # 1. 렌더 제품 경로 가져오기 (문서의 renderProductPath 입력값)
                render_product_obj = rep.create.render_product(annotator.GetPath(), [1, 1], name="Isaac")
                render_product_path = render_product_obj.path
                # 2. 각 환경별 고유 그래프 경로
                graph_path = f"/World/Lidar_ROS_Graph_env_{i}"
                
                # 3. OmniGraph 노드 생성 및 설정
                og.Controller.edit(
                    {"graph_path": graph_path, "evaluator_name": "push"},
                    {
                        og.Controller.Keys.CREATE_NODES: [
                            ("OnTick", "omni.graph.action.OnTick"),
                            # 문서에 명시된 Unique ID 사용
                            ("LidarHelper", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
                        ],
                        og.Controller.Keys.CONNECT: [
                            ("OnTick.outputs:tick", "LidarHelper.inputs:execIn"),
                        ],
                        og.Controller.Keys.SET_VALUES: [
                            # 문서의 Inputs 리스트와 1:1 매칭
                            ("LidarHelper.inputs:renderProductPath", render_product_path),
                            ("LidarHelper.inputs:topicName", f"env_{i}/unitree_go2/lidar/point_cloud"),
                            ("LidarHelper.inputs:frameId", f"base_scan_{i}"),
                            # metadata에 명시된 allowedTokens (laser_scan, point_cloud) 중 선택
                            ("LidarHelper.inputs:type", "point_cloud"), 
                            ("LidarHelper.inputs:fullScan", True), # 전체 스캔 발행 여부
                            ("LidarHelper.inputs:resetSimulationTimeOnStop", True),
                        ],
                    },
                )
                print(f"[Bridge] LiDAR Helper Node (Env {i}) created successfully.")

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

    def pub_ros2_data(self):
        """OmniGraph/Replicator 방식이므로 루프에서 할 일 없음 (에러 방지용 빈 함수)"""
        pass

    def destroy_node(self):
        print("[INFO] ROS2 Bridge Manager Shutting Down...")