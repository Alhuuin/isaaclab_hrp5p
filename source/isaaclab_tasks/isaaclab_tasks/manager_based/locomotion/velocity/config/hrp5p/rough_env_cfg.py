from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

from isaaclab_assets import HRP5P_CFG  # isort: skip

@configclass
class HRP5Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-600.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Lleg_Link5","Rleg_Link5"]),
            "threshold": 0.4,
        },
    )
    joint_extension_penalty = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LKP", "RKP"])}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Lleg_Link5","Rleg_Link5"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["Lleg_Link5","Rleg_Link5"]),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LAP", "LAR", "RAP", "RAR","LKP", "RKP","WP", "WR", "WY","LCY", "LCR","LCP", "RCY", "RCR","RCP"])}
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LCY", "LCR", "RCY", "RCR"])}
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["LCP","RCP"])}
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["LSP", "LSR", "LSY", "LEP", "RSP", "RSR", "RSY", "REP"]
        )}
    )
    # joint_deviation_fingers = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.05,
    #     params={"asset_cfg": SceneEntityCfg(
    #         "robot",
    #         joint_names=[
    #             "LHDY", "LIMP", "LIPIP", "LIDIP", "LMMP", "LMPIP", "LMDIP", "LTMP", "LTPIP", "LTDIP",
    #             "RHDY", "RIMP", "RIPIP", "RIDIP", "RMMP", "RMPIP", "RMDIP", "RTMP", "RTPIP", "RTDIP"
    #         ]
    #     )}
    # )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["WP", "WR", "WY"])}
    )

@configclass
class HRP5RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: HRP5Rewards = HRP5Rewards()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = HRP5P_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Chest_Link2"

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Chest_Link2", "Chest_Link1", "Chest_Link0"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["LCY", "LCP", "LKP", "LAP", "RCY", "RCP", "RKP", "RAP", "LAR", "RAR"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["LCY", "LCP", "LKP", "LAP", "RCY", "RCP", "RKP", "RAP", "LAR", "RAR"]
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = ["Body","Chest_Link2","Rleg_Link2","Lleg_Link2","Larm_Link0","Rarm_Link0",
                                                                          "Larm_Link1","Rarm_Link1","Larm_Link2","Rarm_Link2","Larm_Link3","Rarm_Link3"]
        
"""                 # -- Observations: disable fingers (not necesarry)
        self.observations.policy.enable_dof_pos_vel = True
        self.observations.policy.enable_noise = True
        self.observations.policy.privileged = False

        # Keep only usefull DOFs for locomotion (legs + torso)
        self.observations.policy.dof_names = [
            "LCY", "LCP", "LKP", "LAP", "LAR",
            "RCY", "RCP", "RKP", "RAP", "RAR",
            "WP", "WR", "WY"
        ] """


@configclass
class HRP5RoughEnvCfg_PLAY(HRP5RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

