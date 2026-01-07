import os
import sys
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import hydra
from omegaconf import DictConfig

# 설정 파일 경로 설정 (sim.yaml이 있는 위치)
FILE_PATH = os.path.join(os.path.dirname(__file__), "config")

class Go2MultiOdomTFPublisher(Node):
    def __init__(self, cfg):
        super().__init__('go2_multi_odom_tf_publisher')
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.get_logger().info(f'Initializing TF Publisher for {self.num_envs} environments...')

        # 1. TF 브로드캐스터 초기화
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # 구독자(Subscriber)들을 담아둘 리스트 (GC 방지용)
        self.subscriptions_list = []

        # 2. Static TF 발행: world -> odom_{i} 및 base_link_{i} -> sensors
        self.publish_static_transforms()


        # 2. 환경 갯수만큼 루프를 돌며 설정
        for i in range(self.num_envs):


            # (2) 토픽 구독: env_{i}/unitree_go2/odom
            topic_name = f'/env_{i}/unitree_go2/odom'
            sub = self.create_subscription(
                Odometry,
                topic_name,
                self.odom_callback, # 모든 로봇이 같은 콜백을 공유합니다.
                10
            )
            self.subscriptions_list.append(sub)
            self.get_logger().info(f'Subscribed to: {topic_name}')

    def publish_static_transforms(self):
        static_transforms_list = []
        now = self.get_clock().now().to_msg()
        
        for i in range(self.num_envs):
            # 1. world -> odom_{i} (모든 로봇의 odom 원점은 world 0,0,0으로 고정)
            t_odom = TransformStamped()
            t_odom.header.stamp = now
            t_odom.header.frame_id = 'world'
            t_odom.child_frame_id = f'odom_{i}'
            t_odom.transform.rotation.w = 1.0
            static_transforms_list.append(t_odom)
            self.get_logger().info(f'[Static TF] world -> odom_{i}')


            # 2. base_link_{i} -> front_cam_link_{i}
            t_cam = TransformStamped()
            t_cam.header.stamp = now
            t_cam.header.frame_id = f'base_link_{i}'
            t_cam.child_frame_id = f'front_cam_link_{i}'
            t_cam.transform.translation.x = float(self.cfg.sensor.camera.pos[0])
            t_cam.transform.translation.y = float(self.cfg.sensor.camera.pos[1])
            t_cam.transform.translation.z = float(self.cfg.sensor.camera.pos[2])
            # Isaac Sim (WXYZ) -> ROS 2 (XYZW)
            t_cam.transform.rotation.w = float(self.cfg.sensor.camera.rot[0])
            t_cam.transform.rotation.x = float(self.cfg.sensor.camera.rot[1])
            t_cam.transform.rotation.y = float(self.cfg.sensor.camera.rot[2])
            t_cam.transform.rotation.z = float(self.cfg.sensor.camera.rot[3])
            static_transforms_list.append(t_cam)
            self.get_logger().info(f'[Static TF] base_link_{i} -> front_cam_link_{i}')

            # 3. base_link_{i} -> go2_lidar{i} (bridge 설정에 맞춰 언더바 제외)
            t_lidar = TransformStamped()
            t_lidar.header.stamp = now
            t_lidar.header.frame_id = f'base_link_{i}'
            t_lidar.child_frame_id = f'go2_lidar{i}'
            t_lidar.transform.translation.x = float(self.cfg.sensor.lidar.pos[0])
            t_lidar.transform.translation.y = float(self.cfg.sensor.lidar.pos[1])
            t_lidar.transform.translation.z = float(self.cfg.sensor.lidar.pos[2])
            t_lidar.transform.rotation.w = float(self.cfg.sensor.lidar.rot[0])
            t_lidar.transform.rotation.x = float(self.cfg.sensor.lidar.rot[1])
            t_lidar.transform.rotation.y = float(self.cfg.sensor.lidar.rot[2])
            t_lidar.transform.rotation.z = float(self.cfg.sensor.lidar.rot[3])
            static_transforms_list.append(t_lidar)
            self.get_logger().info(f'[Static TF] base_link_{i} -> go2_lidar{i}')

        self.static_tf_broadcaster.sendTransform(static_transforms_list)
        self.get_logger().info(f'Published {len(static_transforms_list)} static transforms.')

    def odom_callback(self, msg):
        """
        모든 환경의 Odom 메시지를 처리하는 통합 콜백
        Isaac Sim에서 odomFrameId를 odom_{i}, chassisFrameId를 base_link_{i}로
        이미 설정해서 보내주므로, msg의 header 정보를 그대로 사용하면 됩니다.
        """
        t = TransformStamped()

        # 메시지의 헤더(timestamp, frame_id)를 그대로 사용
        # msg.header.frame_id는 이미 'odom_{i}'로 들어옵니다.
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id 
        t.child_frame_id = msg.child_frame_id  

        # Odometry의 pose 정보를 Transform에 복사
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation

        # TF 브로드캐스트 (odom_{i} -> base_link_{i})
        self.tf_broadcaster.sendTransform(t)

@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def main(cfg: DictConfig):
    # ROS 2 초기화
    rclpy.init(args=None)
    
    # Hydra 설정에서 num_envs 가져오기
    try:
        num_envs = cfg.num_envs
        print(f"[Hydra] Loaded configuration. num_envs: {num_envs}")
    except Exception as e:
        print(f"[Error] Failed to read num_envs from config: {e}")
        num_envs = 1 # 기본값

    node = Go2MultiOdomTFPublisher(cfg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()