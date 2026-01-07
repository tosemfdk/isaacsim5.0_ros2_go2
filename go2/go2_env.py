from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort:skip
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.sensors import Camera, CameraCfg



import gymnasium as gym
import yaml, os
import torch
import go2.go2_ctrl as go2_ctrl


@configclass
class Myscene(InteractiveSceneCfg):
    # 지형 정의
    terrain = TerrainImporterCfg(
        prim_path = "/World/ground",
        terrain_type = "plane",
    )

    # 로봇 정의
    go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2")


    front_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Go2/base/front_cam",
        update_period=0.05,                      
        data_types=["rgb"],              
        spawn=sim_utils.PinholeCameraCfg(),             
        width=640,                                       
        height=480,                                      
        offset=CameraCfg.OffsetCfg(
            pos=(0.5, 0.0, 0.1),                         
            rot=(1.0, 0.0, 0.0, 0.0),                    
            convention="world"                             
        ),
    )
    # 센서 정의
    height_scanner = RayCasterCfg(
        prim_path = "{ENV_REGEX_NS}/Go2/base",
        update_period = 0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)), # pattern_cfg
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # 조명 정의
    light = AssetBaseCfg(
        prim_path = "/World/light",
        spawn = sim_utils.DistantLightCfg(intensity=1000.0),
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    velocity_commands = mdp.UniformVelocityCommandCfg(
        asset_name="go2",
        resampling_time_range=(0.0, 0.0),
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="go2", joint_names=[".*"], scale=0.25, use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # Observation terms (order matters)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("go2")})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("go2")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("go2")})
        velocity_commands = ObsTerm(func=go2_ctrl.base_vel_cmd)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("go2")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("go2")})
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # policy group
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # 로봇 초기 위치 및 자세 설정 (Reset)
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "asset_cfg": SceneEntityCfg("go2"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5), # 기본 자세(default joint pos) 주변에서 랜덤화
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("go2"),
        },
    )

@configclass
class RewardsCfg:
    """Configuration for events."""
    pass
@configclass
class TerminationsCfg:
    """Configuration for events."""
    pass
@configclass
class CurriculumCfg:
    """Configuration for events."""
    pass

@configclass
class Go2RLEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the RL environment."""
    # Scene settings
    scene: Myscene = Myscene(num_envs=2, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # dummy settings
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [-4.0, 0.0, 2.0]
        self.viewer.lookat = [0.5, 0.5, 0.0]
        
        # general settings
        self.decimation = 8
        self.episode_length_s = 20.0
        self.is_finite_horizon = False
        self.actions.joint_pos.scale = 0.25

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation 
        self.sim.render.antialiasing_mode = None

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt




def go2_rl_env(env_cfg,cfg):

    env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=env_cfg, render_mode="rgb_array")
    with open(cfg.agent_cfg_path, 'r') as f:
        unitree_go2_rough_cfg = yaml.safe_load(f)
    agent_cfg = RslRlOnPolicyRunnerCfg(**unitree_go2_rough_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    
    model_path = get_checkpoint_path(log_path=os.path.abspath("models"), 
                                run_dir=agent_cfg.load_run, 
                                checkpoint=agent_cfg.load_checkpoint)
    
    ppo_runner.load(model_path)
    
    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(model_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    return env, policy
