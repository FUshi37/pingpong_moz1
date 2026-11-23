# WMJ-V1 交接技术附录与精确索引

整理日期：2026-11-23。与 [主交接文档](WMJ_V1_CODEX_HANDOFF_20261123.md) 一起阅读。
本附录由当前配置、模块AST和a52已保存布局生成；索引存在不表示对应路线已通过实验验收。

## A. 迁移清单与核验边界

- 当前HEAD：`f2706492bd53e8e4cdfc093dcf1d847d21f344da`；当前分支：`main`。
- 主源码/合同候选清单：683文件；独立资产/支持依赖：8331文件，7,257,019,904字节（约6.759GiB）。
- WMJ输出按分配空间计：6,695,542,784bytes；原6.25GiB总额不重置。
- 资产清单也含历史manifest引用的额外支持源码与RMP参考；这些不一定是当前入口的最小依赖。主源码清单只用于显式审核暂存，不能不看diff直接提交共享文件。
- 不含个人密钥/账号环境；已装包版本清单不包含pip下载凭据。原始用户附件在旧个人目录，应按主文档保存的内容/hash核对。
- 扫描覆盖静态导入、源文件字面量、递归JSON引用、完整WMJ输出、冻结V319快照和RMP参考；不承诺自动解析所有运行时动态路径。V324/V319受信父pickle另作只读核验，replay四文件存在且入清单。
- 原a52记录的226个WM源码hash与当前文件一致；Git HEAD前进本身不代表WM源码发生变化。
- `INVENTORY`每个文件包含相对路径、SHA-256、逻辑大小/分配大小、角色和整理时是否Git跟踪。文档及清单自身不做循环自哈希。

| 文件 | 用法 |
| --- | --- |
| [INVENTORY](WMJ_V1_HANDOFF_INVENTORY_20261123.json) | 迁移后逐文件校验；包含源码/资产、环境包版本、旧Git状态 |
| [SOURCE_PATHS](WMJ_V1_HANDOFF_SOURCE_PATHS_20261123.txt) | 审核后供git --pathspec-from-file显式暂存 |
| [ASSET_PATHS](WMJ_V1_HANDOFF_ASSET_PATHS_20261123.txt) | 供rsync --files-from逐文件迁移 |
| [PACKAGES](WMJ_V1_HANDOFF_PACKAGES_20261123.txt) | 已装包版本快照，不是保证跨驱动通用的安装锁文件 |

### A.1 静态扫描尚不能解析的引用

下列14项包含模板占位符、旧失败尝试/已更名输出和历史依赖；**不是14个当前运行已确认缺失项**。不要创建同名假文件使检查通过。当前关键模型、a52结果及配置直接认证输入已核验；将来要重跑某一旧实验时，必须重新追溯它是否真正使用这里的路径。

- `pingpong_controller/outputs/rl_sim/fplc_v319_gpu0_mean0265_completion_20261026_r1/source_snapshot/pingpong_controller/tools/rl_sim/REPLACE.npz`
- `pingpong_controller/outputs/rl_sim/fplc_v319_gpu0_mean0265_completion_20261026_r1/source_snapshot/pingpong_controller/tools/rl_sim/rmp_pd_candidate_stability.npz`
- `pingpong_controller/outputs/rl_sim/record_new3_5_real_failure_rootcause_20260813/formal_gpu1_v152_recovery_replay_from_v146_u7345_nsteps128_online1_20260831/51_record_new3_sim2real_v152_b2_recovery_anchor_replay_60hz.pkl`
- `pingpong_controller/outputs/rl_sim/wmj_v1/history_rollforward_native_20261106_a01.json`
- `pingpong_controller/outputs/rl_sim/wmj_v1/native_audit_a01.json`
- `pingpong_controller/outputs/rl_sim/wmj_v1/native_audit_a03.json`
- `pingpong_controller/outputs/rl_sim/wmj_v1/native_audit_a05.json`
- `pingpong_controller/outputs/rl_sim/wmj_v1/native_model.mjb`
- `pingpong_controller/outputs/rl_sim/wmj_v1/reset_and_surfaces.npz`
- `pingpong_controller/tools/rl_2real/interrupted_u{update:08d}.pkl`
- `pingpong_controller/tools/rl_2real/periodic_u{update:08d}.pkl`
- `pingpong_controller/tools/rl_2real/training_qualified_u{update:08d}.pkl`
- `pingpong_controller/tools/rl_sim/REPLACE.npz`
- `pingpong_controller/tools/rl_sim/rmp_pd_candidate_stability.npz`

## B. 原生认证输入

| 名称 | 路径 | SHA-256 |
| --- | --- | --- |
| v324_config | `pingpong_controller/tools/rl_2real/WMJ_V324_CONFIG.json` | `2c62a5624ef201ace732fa013299e03f4cab2841235795a604bd16865c1ddd0b` |
| v321_release | `pingpong_controller/outputs/rl_sim/releases/fplc_v321_s4_u2097_20261029_r1/release.json` | `40211fbd02b966e8709d2d94e5391fedff390d18d4953659344ec972a224bbac` |
| execution_manifest | `pingpong_controller/tools/rl_sim/execution_versions/mjx_topology_execution_v2/manifest.json` | `2cc5730b85e46f4da56a688df445fa963529efe6f033e3b37bd905d2e9bb9a47` |
| execution_source | `pingpong_controller/tools/rl_sim/mjx_topology_execution.py` | `3d1bed370804d4ec5c2a07bffbdadbb7c250b91f67ebda1d0801e122b2d416f0` |
| projection_source | `pingpong_controller/tools/rl_sim/cartesian_pose_action_projection_jax.py` | `0f43f1f4d05a410f3fd066d7776980bb075a97f74566f00249aaadbea553929e` |
| xml | `pingpong_controller/outputs/rl_sim/fplc_v319_gpu0_mean0265_completion_20261026_r1/source_snapshot/pingpong_controller/tools/rl_sim/moz1_pd_1ms_v98.xml` | `ace35229f4d643410a95481719c962e3b8ad172c2eba5901ee3d3e76e966ad8a` |
| v324_complete_env | `pingpong_controller/outputs/rl_sim/v323_v324_training_only_20261105_r1/execution_pair_audit_v324.json` | `9dd383c08720cef529d8695e00f20aee7deb940a76d046980727a8e59f182bf3` |

## C. 当前基础合同的完整字段摘录

这些是基础配置，私有候选的覆盖见主文档和各plan。为了保持早期checkpoint兼容，不把所有历史块强行改成最新候选。

### C.identity

```json
{
  "project": "WMJ",
  "training_version": "V1",
  "profile": "wmj_v1",
  "config_rev": "r01",
  "automatic_major_version_changes": false,
  "model_revision_role": "checkpoint_refresh_counter_only",
  "status": "implementation_in_progress_not_qualified"
}
```

### C.training_protocol

```json
{
  "path": "pingpong_controller/tools/rl_2real/WMJ_TRAINING_PROTOCOL.md",
  "sha256": "c3cd399b8e0ff8fd86369befb80eb9eebd55d49c28a05de0fbb778bf748bf1fc",
  "enabled": true,
  "id": "wmj_training_protocol",
  "version": "1.0",
  "overrides": [
    {
      "clause": "research_attachment/observation_noise_schedule",
      "value": "S01-S04 zero noise; S05 full V324 AR1",
      "authority": "user_QA_2026-11-05"
    },
    {
      "clause": "research_attachment/course",
      "value": "append S05 AR1 final consolidation without changing WMJ-V1",
      "authority": "user_QA_2026-11-05"
    },
    {
      "clause": "research_attachment/observation_history",
      "value": "969D actor: preserved73D prefix plus64 ball-indexed14D paired-history rows spanning640ms",
      "authority": "user_QA_2026-11-05"
    },
    {
      "clause": "research_attachment/initialization_geometry",
      "value": "offline only: convex broad phase plus original triangle/containment refinement; UNKNOWN retained; no runtime scans or added clearance",
      "authority": "user_QA_2026-11-06"
    },
    {
      "clause": "research_attachment/robot_execution_state",
      "value": "47->128->128->26 execution proxy: predict RightArm-0..6 and LegWaist-0..5 q/dq; policy remains969D with7 actions; full-chain FK with true pose/twist supervision",
      "authority": "user_recommended_plan_approval_2026-11-06"
    }
  ]
}
```

### C.components

```json
{
  "CONTACT": {
    "id": "ball_racket_contact_real11_static_1ms_v1",
    "configuration_ref": "#/physics/contact",
    "evidence": {
      "path": "pingpong_controller/tools/rl_2real/BALL_RACKET_CONTACT_REAL11_STATIC_1MS_V1_REPORT.md",
      "sha256": "4cc5f36b59d04a7e0a82cdda54269047bfd83ec8d4879c2365dfbd8df1a00829"
    }
  },
  "EXEC": {
    "id": "mjx_topology_execution_v2",
    "configuration_ref": "#/execution"
  },
  "COURSE": {
    "id": "wmj_course_v1",
    "configuration_ref": "#/curriculum"
  },
  "VERIFY": {
    "id": "wmj_verification_v1",
    "configuration_ref": "#/verification"
  }
}
```

### C.lineage

```json
{
  "actor_source": {
    "path": "pingpong_controller/outputs/rl_sim/releases/fplc_v321_s4_u2097_20261029_r1/policy.pkl",
    "sha256": "889eb976b18567d607ab9f86f35457c89f901e2ecf221b9391128cb471f0d001",
    "release_id": "FPLC-V321-S4-U2097-20261029-R1",
    "true_step": 1609695232,
    "update": 2097,
    "adam_t": 402875
  },
  "actor_migration": "copy_72_rows_zero_all_897_added_input_rows",
  "actor_optimizer": "new",
  "critic": "new_privileged_value_network",
  "critic_optimizer": "new",
  "world_model_optimizer": "new",
  "source_simulator": "authenticated_V324_native_stack",
  "reference_commit": "d0292f2e76ce9c5e3ae98a882be04301b909b659",
  "audit_worktree_commit": "a1c25be2566e82faee55431e73a0861c6adba2f8",
  "release_hardware_claim": "user_reported_30_minutes; no new hardware test"
}
```

### C.policy

```json
{
  "actor_obs_dim": 969,
  "representation": "direct_history_mlp",
  "representation_candidates": [
    {
      "id": "prefix73_temporal_history128_structured_physics",
      "status": "candidate_only",
      "requires_user_confirmation": true
    },
    {
      "id": "latent_world_model_and_policy",
      "status": "candidate_only",
      "requires_user_confirmation": true
    }
  ],
  "action_dim": 7,
  "hidden_sizes": [
    256,
    256
  ],
  "hidden_activation": "tanh",
  "output_activation": "linear",
  "normalization": "none",
  "raw_action_clip": [
    -1.0,
    1.0
  ],
  "observation_schema": "wmj_v1_prefix73_paired_ball_racket_640ms",
  "goal_z_encoding": {
    "center_m": 1.425,
    "scale_m": 0.1
  },
  "goal_xy_slice": [
    69,
    71
  ],
  "goal_z_slice": [
    72,
    73
  ],
  "previous_action_semantics": "previous_accepted_projected_normalized_action",
  "actor_forbidden_inputs": [
    "true_ball_velocity",
    "ball_spin",
    "future_contact",
    "uncertainty",
    "privileged_DR"
  ],
  "history_slice": [
    73,
    969
  ]
}
```

### C.control

```json
{
  "policy_dt_s": 0.005,
  "physics_dt_s": 0.001,
  "substeps": 5,
  "action_semantics": "persistent_measured_feedback_QVEL",
  "simulation_base_mode": "aligned_fixed",
  "joint_order": [
    "RightArm-0",
    "RightArm-1",
    "RightArm-2",
    "RightArm-3",
    "RightArm-4",
    "RightArm-5",
    "RightArm-6"
  ],
  "native_parameters": {
    "right_arm_pd_profile": "recovered_rmp_rmpmd_v2",
    "arm_vel_limit_deg_s": [
      210.0,
      210.0,
      240.0,
      240.0,
      300.0,
      300.0,
      300.0
    ],
    "arm_acc_limit_deg_s2": [
      1300.0,
      1300.0,
      1800.0,
      3000.0,
      3000.0,
      3000.0,
      3000.0
    ],
    "recovered_rmp_motion_mode": true,
    "recovered_rmp_period_s": 0.001,
    "recovered_rmp_direct_qvel_target": true,
    "recovered_rmp_qvel_target_lead_s": 0.0,
    "recovered_rmp_bounded_qvel_reference": true,
    "recovered_rmp_qvel_reference_error_horizon_s_per_joint": [
      10.0,
      10.0,
      10.0,
      10.0,
      10.0,
      10.0,
      10.0
    ],
    "recovered_rmp_jnt_kp_scale_per_joint": [
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "recovered_rmp_jnt_kd_scale_per_joint": [
      1.0,
      1.0,
      0.92,
      1.0,
      1.0,
      1.0,
      1.0
    ],
    "recovered_rmp_dof_damping_scale_per_joint": [
      0.6084,
      0.9984,
      0.9984,
      1.92,
      0.58,
      1.17,
      0.78
    ],
    "recovered_rmp_dof_armature_scale_per_joint": [
      0.4524,
      1.65,
      0.92,
      1.22,
      0.58,
      0.5336,
      0.58
    ],
    "recovered_rmp_target_filter_length": 10,
    "recovered_rmp_output_filter_length": 35,
    "recovered_rmp_target_update_delay_steps": 3,
    "recovered_rmp_output_delay_steps_per_joint": [
      18,
      19,
      19,
      18,
      18,
      18,
      18
    ],
    "recovered_rmp_joint_velocity_feedforward": 0.5,
    "recovered_rmp_joint_acceleration_feedforward": 0.0,
    "recovered_rmp_estimator_joint_process_var": 50.0,
    "recovered_rmp_estimator_joint_measure_var": 0.0001,
    "recovered_rmp_acceleration_weight": 0.001,
    "recovered_rmp_use_safety_bounds": true,
    "recovered_rmp_use_jerk_limit": false
  },
  "new_workspace_overrides_ref": "#/domain",
  "robot_deployment_enabled": false
}
```

### C.physics

```json
{
  "xml": {
    "path": "pingpong_controller/outputs/rl_sim/fplc_v319_gpu0_mean0265_completion_20261026_r1/source_snapshot/pingpong_controller/tools/rl_sim/moz1_pd_1ms_v98.xml",
    "sha256": "ace35229f4d643410a95481719c962e3b8ad172c2eba5901ee3d3e76e966ad8a"
  },
  "ball_mass_kg": 0.004,
  "ball_mass_range_kg": [
    0.004,
    0.004
  ],
  "ball_radius_m": 0.02,
  "compiled_geometry_authoritative": true,
  "gravity_m_s2": [
    0.0,
    0.0,
    -9.81
  ],
  "drag_coefficient_m_inv": 0.0,
  "collision_outcome_residual_enabled": false,
  "contact": {
    "ball_racket_collision_geometry": "finite_mesh16",
    "ball_racket_pair_override_enabled": true,
    "ball_racket_pair_solref": [
      0.006836,
      0.228086
    ],
    "ball_racket_pair_solreffriction": [
      0.0,
      0.0
    ],
    "ball_racket_pair_solimp": [
      0.04,
      0.7,
      0.014298,
      0.04,
      8.021856
    ],
    "ball_racket_pair_friction": [
      0.13268182450948787,
      0.11820649851379933,
      0.0001544955787453423,
      2.013852101985789e-05,
      2.013852101985789e-05
    ],
    "ball_racket_pair_margin_m": 0.0,
    "ball_racket_pair_gap_m": 0.0,
    "ball_racket_pair_condim": 3,
    "ball_racket_pair_tie_tangent_friction_dr": false,
    "ball_racket_contact_boundary_mode": "robot_dynamic",
    "fixed_plane_v98_coarse_contact": true,
    "compiled_racket_geom_offset_along_normal_m": 0.003,
    "calibration_population": "real11_static_first_rebound",
    "calibration_case_count": 11,
    "independent_dynamic_held_out_certification": false
  },
  "integrator": "implicitfast",
  "solver_iterations": 100,
  "line_search_iterations": 50,
  "contact_DR_enabled": false,
  "new_hardware_calibration_claim": false
}
```

### C.domain

```json
{
  "task_goal_xyz_lower_m": [
    -0.38,
    -0.75,
    1.3
  ],
  "task_goal_xyz_upper_m": [
    0.38,
    -0.28,
    1.45
  ],
  "racket_xyz_lower_m": [
    -0.38,
    -0.75,
    1.0
  ],
  "racket_xyz_upper_m": [
    0.38,
    -0.28,
    1.15
  ],
  "roll_pitch_limit_deg": 2.0,
  "view_xy_lower_m": [
    -0.38,
    -0.75
  ],
  "view_xy_upper_m": [
    0.38,
    -0.28
  ],
  "view_hardware_certified": false,
  "view_status": "user_authorized_simulation_assumption",
  "ball_z_physical_bounds_m": [
    0.98,
    1.5
  ],
  "domain_admission": "compiled_geometry_multistart_IK_joint_self_collision_and_velocity_audit",
  "self_collision_audit": {
    "scope": "offline_initialization_only",
    "method": "convex_broad_phase_original_triangle_and_containment_refinement",
    "runtime_step_enabled": false,
    "runtime_reset_scan_enabled": false,
    "reset_source": "preaudited_seed_bank",
    "additional_clearance_m": 0.0,
    "surface_tolerance_m": 1e-09,
    "ambiguous_result": "UNKNOWN",
    "repair_or_fill_mesh_holes": false,
    "trajectory_safety_certified": false,
    "preparation_wall_time_in_total": true
  },
  "unknown_reachability": "retain_separate_not_policy_selected",
  "task_goal_separate_from_workspace_origin": true
}
```

### C.observation

```json
{
  "rate_hz": 90.0,
  "history_policy_ticks": 8,
  "fractional_rate_scheduler": true,
  "zero_order_hold": true,
  "age_scale_s": 0.25,
  "reset_older_history_age_s": 1.0,
  "ar1": {
    "enabled_by_stage": true,
    "mean_m": [
      4.8e-05,
      7.2e-05,
      -0.000227
    ],
    "std_m": [
      0.001451,
      0.001418,
      0.00297
    ],
    "rho_at_90hz": [
      0.935,
      0.907,
      0.79
    ],
    "advance_only_on_fresh_sample": true
  },
  "other_noise_enabled": false,
  "world_transition": {
    "kind": "known_camera_and_history_roll_forward",
    "new_state_source": "predicted_physics_then_observation_pipeline",
    "re_predict_history": false,
    "future_teacher_observations": false,
    "legacy_short_history_rate_hz": 200.0,
    "fresh_long_history_rate_hz": 90.0,
    "hold_between_fresh_samples": true,
    "past_measurements_and_noise": "preserve",
    "long_history_age": "query_time_minus_source_time",
    "short_history_age": "preserve_native_age_at_each_policy_tick",
    "racket_pose_source": "FK_of_same_timestamp_measured_or_predicted_q",
    "seed": "complete_camera_rng_phase_feedback_and_paired_history",
    "gradient": "retain_within_window_cut_at_physical_reset_or_window_boundary"
  },
  "paired_history": {
    "window_s": 0.64,
    "ball_sample_slots": 64,
    "features_per_sample": 14,
    "row_layout": {
      "ball_position_world_m": [
        0,
        3
      ],
      "racket_site_position_world_m": [
        3,
        6
      ],
      "racket_rotation_first_two_columns": [
        6,
        12
      ],
      "sample_age_div_0_640_s": [
        12,
        13
      ],
      "valid_same_timestamp_pair": [
        13,
        14
      ]
    },
    "ordering": "newest_fresh_ball_sample_first",
    "append_only_fresh_ball_samples": true,
    "racket_dense_actor_history": false,
    "joint_action_actor_history": false,
    "pose_alignment": "ball_source_timestamp_causal_encoder_interpolation_FK",
    "source_time_units": "episode_relative_us",
    "internal_encoder_cache_samples": 129,
    "internal_encoder_cache_is_actor_input": false,
    "max_encoder_interpolation_gap_us": 5000,
    "invalid_or_expired_rows": "zero_with_valid_mask_false",
    "physical_reset": "clear_with_valid_masks",
    "goal_switch": "preserve",
    "latent_compression_enabled": false,
    "storage": "continuous_samples_and_window_indices_not_duplicated_per_transition"
  }
}
```

### C.task

```json
{
  "hits_per_goal": 16,
  "latest_stability_start_hit": 3,
  "required_stable_apexes": 10,
  "xy_tolerance_m": 0.02,
  "z_tolerance_m": 0.02,
  "vxy_limit_m_s": 0.05,
  "switch_after": "actual_apex_of_16th_attributed_contact",
  "latch_goal_at": "physical_valid_controlled_release",
  "exclude_initialization_launch": true,
  "remaining_hits_require_survival": true,
  "physical_reset_on_goal_switch": false,
  "evaluation_apex": "actual_trajectory",
  "reward_apex": "ballistic_prediction_after_valid_release_allowed"
}
```

### C.collection

```json
{
  "n_envs": 1024,
  "initialization_atlas": {
    "path": "pingpong_controller/outputs/rl_sim/wmj_v1/domain_atlas_20261106_a01.json",
    "sha256": "2244c5c6fa6fad10cf83eacb02412f68f4f168bb039a303d19c4f99b62ac466d"
  },
  "compiled_model_cache": {
    "path": "pingpong_controller/outputs/rl_sim/wmj_v1/collision_geometry_a02/native_model.mjb",
    "sha256": "65b41fd91a84521e8762b67fbd920da7956c81e45394909a88602c6eab14d611"
  },
  "directed_conditions": {
    "incoming_vertical_velocity_m_s": [
      -3.5,
      -1.0
    ],
    "incoming_horizontal_speed_max_m_s": 0.5,
    "time_to_contact_s": [
      0.08,
      0.16
    ],
    "initial_ball_angular_velocity_rad_s": [
      0.0,
      0.0,
      0.0
    ],
    "sample_initial_spin": false,
    "preserve_collision_induced_spin": true,
    "offset_classes": [
      "center",
      "off_center",
      "edge",
      "near_miss"
    ],
    "keep_actual_failed_outcomes": true,
    "scope": "simulation_initial_conditions_not_contact_parameters_or_hardware_envelope"
  },
  "steps_per_round": 64,
  "generator_execution": {
    "warmup_controls": 128,
    "warmup_initial_zero_controls": 64,
    "warmup_normalized_amplitude_range": [
      0.005,
      0.025
    ],
    "trial_controls": 128,
    "identification_normalized_amplitude_range": [
      0.02,
      0.25
    ],
    "boundary_normalized_amplitude_range": [
      0.5,
      1.5
    ],
    "closed_loop_teacher": "pinned_V321_on_global_trials_no_policy_updates",
    "preview": "actual_V324_control_chain_before_ball_initialization",
    "out_of_task_bounds": "keep_as_physics_data_label_and_exclude_from_initial_task_seeds",
    "robot_identification_remote_ball": true
  },
  "transitions_per_round": 65536,
  "budget_rounds": [
    20,
    40,
    80
  ],
  "budget_is_cumulative": true,
  "generator_fractions": {
    "robot_identification": 0.25,
    "directed_contact": 0.4,
    "closed_loop": 0.2,
    "boundaries_failure": 0.15
  },
  "old_fixed_point_fraction_max": 0.1,
  "global_domain_from_first_round": true,
  "regular_rate_hz": 200,
  "contact_rate_hz": 1000,
  "contact_pre_ms": 25,
  "contact_post_ms": 25,
  "state_checkpoint_interval_control_steps": 128,
  "save_complete_seed_bank": true,
  "incompatible_buffer_indices": "reject_or_phase_group_never_blind_concatenate",
  "normalize_using": "train_only",
  "split": {
    "id": "wmj_group_split_v1",
    "fractions": [
      0.8,
      0.1,
      0.1
    ],
    "group_keys": [
      "root_seed_family",
      "session",
      "episode"
    ],
    "stable_append": true,
    "final_test_for_active_query": false
  }
}
```

### C.samplers

```json
{
  "P_collect": {
    "global_stratified_fraction": 0.5,
    "weak_region_fraction": 0.5,
    "uncertainty": "only_if_calibrated_against_validation_error",
    "test_set_access": false
  },
  "P_model": {
    "batch_windows": 256,
    "valid_contact": 128,
    "miss_failure": 64,
    "robot_identification": 38,
    "free_flight": 26,
    "exclusive_primary_buckets": true,
    "balance_independent_events": true,
    "shortage": "record_reallocation_or_collect_more"
  },
  "P_start": {
    "precontact": 128,
    "goal_switch": 64,
    "recoverable_disturbance": 64,
    "respect_control_latency": true,
    "course_counts": {
      "S01": {
        "precontact": 256,
        "goal_switch": 0,
        "recoverable_disturbance": 0
      },
      "S02": {
        "precontact": 192,
        "goal_switch": 64,
        "recoverable_disturbance": 0
      },
      "S03": {
        "precontact": 192,
        "goal_switch": 64,
        "recoverable_disturbance": 0
      },
      "S04": {
        "precontact": 128,
        "goal_switch": 64,
        "recoverable_disturbance": 64
      },
      "S05": {
        "precontact": 128,
        "goal_switch": 64,
        "recoverable_disturbance": 64
      }
    },
    "missing_enabled_category": "reassign_to_precontact_and_record_until_causal_complete_seed_available",
    "invent_task_history": false,
    "bind_seed_to_model_and_noise": true
  },
  "P_goal": {
    "global_uniform": 0.5,
    "learning_frontier": 0.3,
    "mastered_review": 0.2,
    "frontier_success_interval": [
      0.2,
      0.8
    ],
    "fixed_evaluation_distribution": true
  }
}
```

### C.imagination

```json
{
  "logical_envs": 1024,
  "persistent_envs": 768,
  "branch_envs": 256,
  "horizon": 64,
  "microbatch": 256,
  "episode_limit_s": 30.0,
  "gamma_5ms": 0.9995,
  "lambda": 0.99,
  "actor_lr": 1e-05,
  "actor_lr_candidates": [
    1e-05,
    3e-05,
    0.0001
  ],
  "critic_lr": 0.0001,
  "global_gradient_clip": 1.0,
  "exploration_std": 0.03,
  "critic_updates_per_actor": 4,
  "critic_batch": 8192,
  "critic_microbatch": 2048,
  "target_critic_ema": 0.01,
  "actor_training_source": "imagination",
  "critic_training_source": "imagination",
  "freeze_per_accumulation": [
    "actor_params",
    "wm_params",
    "target_critic_params",
    "normalization"
  ],
  "detach_between_windows": true,
  "reset_on_goal_switch": false,
  "reinitialize_pool_on_model_refresh": true
}
```

### C.active_query

```json
{
  "diagnose_every_actor_updates": 200,
  "n_envs": 1024,
  "steps": 64,
  "rounds": 2,
  "new_transitions": 131072,
  "refit_old_fraction": 0.8,
  "refit_new_fraction": 0.2,
  "max_updates_per_member": 2000,
  "retain_all_data": true,
  "critic_adaptation_before_actor": true,
  "model_truncation_distinct_from_physical_failure": true,
  "unreliable_bootstrap": false
}
```

### C.experiments

```json
{
  "methods": {
    "B0": "real_MJX_task_PPO_baseline_only",
    "B1": "imagined_PPO",
    "W0": "differentiable_imagined_actor_critic",
    "W1": "W0_plus_event_and_action_difference_supervision"
  },
  "main_method": "W1",
  "data_ablations": [
    "local_only",
    "equal_size_global",
    "global_plus_active"
  ],
  "training_ablations": [
    "frozen_world_model",
    "low_frequency_refresh",
    "persistent_pools",
    "all_short_resets"
  ],
  "full_multiseed_paper_study_in_this_run": false
}
```

### C.verification

```json
{
  "primary_metric": "T90_on_fixed_global_goal_and_transition_distribution",
  "target_success_rate": 0.9,
  "include_initial_collection_and_model_training_in_time": true,
  "report_reuse_incremental_time": true,
  "require_componentwise_errors": true,
  "final_test_not_used_for_tuning": true,
  "gpu_timer_block_until_ready": true,
  "compile_and_steady_timing_separate": true,
  "statuses": [
    "PASS",
    "FAIL",
    "INCOMPLETE",
    "NOT_RUN"
  ],
  "hardware_test_status": "NOT_RUN"
}
```

### C.resources

```json
{
  "output_root": "pingpong_controller/outputs/rl_sim/wmj_v1",
  "max_new_bytes": 6710886400,
  "min_free_bytes": 8589934592,
  "max_run_wall_s": 43200,
  "python": "/home/yangzhe/miniconda3/envs/pingpong/bin/python",
  "allowed_gpu_uuids": [
    "GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4",
    "GPU-e74458ec-002a-1bac-e0be-d0a5713b661e"
  ],
  "gpu_memory_is_not_pooled": true,
  "preallocate": false,
  "max_gpu_temperature_c": 78,
  "memory_high_gib": 10.0,
  "memory_max_gib": 11.0,
  "memory_stop_gib": 9.7,
  "psi_stop_policy": {
    "mode": "wmj_v1_conditional",
    "scope_at_least_gib": 8.0,
    "host_available_at_most_gib": 8.0,
    "otherwise": "log_warning"
  },
  "host_stop_available_gib": 6.0,
  "startup_min_host_available_gib": 13.5,
  "stop_and_ask_on_storage_exhaustion": true,
  "delete_existing_evidence": false
}
```

### C.checkpointing

```json
{
  "atomic": true,
  "overwrite_existing_run": false,
  "save": [
    "wm_params",
    "actor_params",
    "critic_params",
    "all_optimizers",
    "target_critic",
    "RNG",
    "normalization",
    "persistent_pool",
    "task_state",
    "model_revision",
    "source_hashes",
    "config_hash",
    "geometry_hash",
    "split_registry",
    "transition_counts",
    "stage_and_wall_times"
  ],
  "protected": [
    "latest_resume",
    "best_candidate",
    "stage_boundary",
    "qualified",
    "referenced_parent"
  ],
  "automatic_pruning": false
}
```

### C.differentiability_requirement

```json
{
  "mandatory": true,
  "authorized_by": "user_20261112",
  "method": "short_horizon_pathwise_actor_critic",
  "continuous_state_action_derivatives_required": true,
  "freeze_world_weights_not_input_gradients": true,
  "freeze_target_value_weights_not_input_gradients": true,
  "detach_critic_targets": true,
  "detach_at_reset_and_window_boundary": true,
  "status": "implemented_piecewise_path_gradient_tail_and_end_to_end_speedup_not_qualified",
  "global_smoothness_claim": false,
  "required_evidence": [
    "same_mode_finite_differences",
    "boundary_action_outcome_direction",
    "nonzero_finite_actor_gradients",
    "complete_update_and_total_time"
  ],
  "reference": "https://arxiv.org/abs/2610.28480v2"
}
```

### C.world_model 核心

```json
{
  "initial_members": 1,
  "comparison_members": [
    1,
    3
  ],
  "member_fixed_per_episode": true,
  "goal_in_physics_network": false,
  "robot": {
    "base_input_dim": 47,
    "hidden_activation": "tanh",
    "output_activation": "linear",
    "residual_baseline": "q_plus_dq_dt_and_dq",
    "normalization": "train_partition_statistics_only",
    "hidden_sizes": [
      128,
      128
    ],
    "output_dim": 26,
    "output": "arm_delta_q7_delta_dq7_support_delta_q6_delta_dq6",
    "input": "arm_q7_dq7_delayed_q_error7_dq7_ddq7_support_q6_dq6",
    "state_joint_order": [
      "RightArm-0",
      "RightArm-1",
      "RightArm-2",
      "RightArm-3",
      "RightArm-4",
      "RightArm-5",
      "RightArm-6",
      "LegWaist-0",
      "LegWaist-1",
      "LegWaist-2",
      "LegWaist-3",
      "LegWaist-4",
      "LegWaist-5"
    ],
    "support_commands": "unchanged_native_constant_hold_targets",
    "actor_dimension": 969,
    "action_dimension": 7,
    "deployment_execution_proxy_required": false,
    "legacy_arm_only_component": "35->128->128->14 diagnostic retained; cannot qualify full-chain world model",
    "qualification": "requires_native_1ms_support_state_and_racket_pose_twist_labels"
  },
  "reference_substeps": 5,
  "reference_dt_s": 0.001,
  "macro_step_candidate_dt_s": 0.005,
  "contact_compute_dtype": "float32",
  "optimizer": {
    "name": "Adam",
    "lr": 0.0003,
    "max_initial_updates": 20000
  },
  "window_steps": [
    8,
    16,
    32,
    64
  ],
  "window_batch": 256,
  "autoregressive": true,
  "action_difference_epsilon": [
    0.005,
    0.01,
    0.02
  ],
  "action_difference_duration_ms": [
    10,
    20
  ],
  "second_order_jacobian_loss": false,
  "readiness_reference_p95": {
    "precontact_racket_position_m": 0.003,
    "normal_speed_m_s": 0.05,
    "outgoing_velocity_m_s": 0.08,
    "next_apex_m": 0.01
  },
  "evaluate_horizons": [
    1,
    8,
    32,
    64,
    128
  ]
}
```

接触/辅助历次比较树非常长，保留于CFG原文：`world_model.contact`、`world_model.solver_acceleration_auxiliary`。主文档第6节给出当前实际组合与关闭的残差。

## D. Critic 2602维精确布局

来源：a52 `course_critic_layout`保存的71字段；start/end为左闭右开，除数按原报告登记。先做该编码，再输入MLP；下面的组别仅供诊断，没有重排数组。

| 字段 | start:end | shape | divisor | 单位/语义 |
| --- | --- | --- | --- | --- |
| `actor_observation` | 0:969 | `[969]` | `1.0` | existing_actor_encoding |
| `joint_q` | 969:982 | `[13]` | `3.141592653589793` | rad |
| `joint_dq` | 982:995 | `[13]` | `5.0` | rad/s |
| `ball_position` | 995:998 | `[3]` | `1.0` | m |
| `ball_velocity` | 998:1001 | `[3]` | `3.0` | m/s |
| `ball_spin` | 1001:1004 | `[3]` | `30.0` | rad/s |
| `ball_contact` | 1004:1005 | `[]` | `1.0` | bool |
| `ball_contact_age` | 1005:1006 | `[]` | `0.02` | s |
| `reference_q` | 1006:1013 | `[7]` | `3.141592653589793` | rad |
| `previous_accepted_action` | 1013:1020 | `[7]` | `1.0` | normalized_action |
| `rmp.q` | 1020:1027 | `[7]` | `3.141592653589793` | rad |
| `rmp.qd` | 1027:1034 | `[7]` | `5.0` | rad/s |
| `rmp.qdd` | 1034:1041 | `[7]` | `50.0` | rad/s2 |
| `rmp.command_q` | 1041:1048 | `[7]` | `3.141592653589793` | rad |
| `rmp.command_qd` | 1048:1055 | `[7]` | `5.0` | rad/s |
| `rmp.command_qdd` | 1055:1062 | `[7]` | `50.0` | rad/s2 |
| `rmp.accepted_target` | 1062:1069 | `[7]` | `3.141592653589793` | rad |
| `rmp.ticks_since_target_update` | 1069:1070 | `[]` | `5.0` | 1ms_ticks |
| `rmp.target_x` | 1070:1091 | `[7,3]` | `[3.1415927410125732,5.0,50.0]` | q_dq_ddq |
| `rmp.target_covariance` | 1091:1154 | `[7,3,3]` | `[[9.86960506439209,15.707963943481445,157.0796356201172],[15.707963943481445,25.0,250.0],[157.0796356201172,250.0,2500.0]]` | q_dq_ddq_covariance |
| `rmp.target_q_buffer` | 1154:1224 | `[10,7]` | `3.141592653589793` | rad |
| `rmp.target_qd_buffer` | 1224:1294 | `[10,7]` | `5.0` | rad/s |
| `rmp.target_qdd_buffer` | 1294:1364 | `[10,7]` | `50.0` | rad/s2 |
| `rmp.target_buffer_index` | 1364:1365 | `[]` | `10` | ring_index |
| `rmp.output_q_buffer` | 1365:1610 | `[35,7]` | `3.141592653589793` | rad |
| `rmp.output_qd_buffer` | 1610:1855 | `[35,7]` | `5.0` | rad/s |
| `rmp.output_qdd_buffer` | 1855:2100 | `[35,7]` | `50.0` | rad/s2 |
| `rmp.output_buffer_index` | 2100:2101 | `[]` | `35` | ring_index |
| `output_delay.q` | 2101:2234 | `[19,7]` | `3.141592653589793` | rad |
| `output_delay.dq` | 2234:2367 | `[19,7]` | `5.0` | rad/s |
| `output_delay.ddq` | 2367:2500 | `[19,7]` | `50.0` | rad/s2 |
| `context.acceleration_weight` | 2500:2507 | `[7]` | `0.001` | native_parameter |
| `context.estimator_joint_measure_var` | 2507:2514 | `[7]` | `0.0001` | native_parameter |
| `context.estimator_joint_process_var` | 2514:2521 | `[7]` | `50.0` | native_parameter |
| `context.jnt_kd` | 2521:2528 | `[7]` | `10000.0` | native_parameter |
| `context.jnt_kp` | 2528:2535 | `[7]` | `100000.0` | native_parameter |
| `context.joint_acceleration_feedforward` | 2535:2542 | `[7]` | `1.0` | native_parameter |
| `context.joint_velocity_feedforward` | 2542:2549 | `[7]` | `1.0` | native_parameter |
| `context.target_filter_length` | 2549:2556 | `[7]` | `10.0` | native_parameter |
| `context.action_scale` | 2556:2557 | `[]` | `1.0` | multiplier |
| `context.output_delay_steps` | 2557:2564 | `[7]` | `20.0` | 1ms_ticks |
| `camera.phase` | 2564:2565 | `[]` | `20.0` | 200Hz_phase |
| `camera.sample_age` | 2565:2566 | `[]` | `3.0` | 200Hz_ticks |
| `camera.ar1_memory` | 2566:2569 | `[3]` | `0.003` | m |
| `episode.elapsed` | 2569:2570 | `[]` | `6000.0` | 200Hz_ticks |
| `events.contact_active` | 2570:2571 | `[]` | `1.0` | event_state |
| `events.impact_id` | 2571:2572 | `[]` | `64.0` | event_state |
| `events.normal_impulse_n_s` | 2572:2573 | `[]` | `0.02` | event_state |
| `events.contact_duration_steps` | 2573:2574 | `[]` | `20.0` | event_state |
| `events.controlled_contact` | 2574:2575 | `[]` | `1.0` | event_state |
| `task.goal_xyz` | 2575:2578 | `[3]` | `1.0` | m |
| `task.goal_id` | 2578:2579 | `[]` | `64.0` | task_state |
| `task.hit_count` | 2579:2580 | `[]` | `16.0` | task_state |
| `task.apex_count` | 2580:2581 | `[]` | `16.0` | task_state |
| `task.last_impact_id` | 2581:2582 | `[]` | `64.0` | task_state |
| `task.flight_goal_xyz` | 2582:2585 | `[3]` | `1.0` | m |
| `task.flight_goal_id` | 2585:2586 | `[]` | `64.0` | task_state |
| `task.flight_hit_index` | 2586:2587 | `[]` | `16.0` | task_state |
| `task.awaiting_apex` | 2587:2588 | `[]` | `1.0` | task_state |
| `task.arrival_hit` | 2588:2589 | `[]` | `16.0` | task_state |
| `task.stable_streak` | 2589:2590 | `[]` | `16.0` | task_state |
| `task.max_stable_streak` | 2590:2591 | `[]` | `16.0` | task_state |
| `task.stability_start_hit` | 2591:2592 | `[]` | `16.0` | task_state |
| `task.stability_qualified` | 2592:2593 | `[]` | `1.0` | task_state |
| `task.failed` | 2593:2594 | `[]` | `1.0` | task_state |
| `task.physical_failure` | 2594:2595 | `[]` | `1.0` | task_state |
| `task.finished` | 2595:2596 | `[]` | `1.0` | task_state |
| `sequence.completed_goals` | 2596:2597 | `[]` | `4.0` | goals |
| `sequence.pulse_used` | 2597:2598 | `[]` | `1.0` | bool |
| `pulse.remaining_steps` | 2598:2599 | `[]` | `20.0` | 1ms_ticks |
| `pulse.force_world_n` | 2599:2602 | `[3]` | `0.02` | N |

## E. CFG中的逐阶段实验状态与证据导航

该表原样保留status文本，可能描述该块形成时的历史状态；解释其与后续阶段关系见主文档1.1。这里只列身份与证据，不授予重跑权限。

| 配置块 | 状态 | authorized/default_enabled | 主要证据/plan |
| --- | --- | --- | --- |
| `imagination_implementation` | mixed_precision_candidate_frozen_validation_complete_awaiting_user_QA | 未定义/False | 见CFG子字段 |
| `frozen_evaluation` | 见对应配置 | 未定义/未定义 | 见CFG子字段 |
| `frozen_evaluation_artifacts` | 见对应配置 | 未定义/未定义 | 见CFG子字段 |
| `frozen_evaluation_numerics` | 见对应配置 | 未定义/未定义 | 见CFG子字段 |
| `offline_initialization_search` | 见对应配置 | 未定义/未定义 | 见CFG子字段 |
| `training_seed_bank` | 1024_unique_complete_states_verified_no_parameter_updates | 未定义/未定义 | bank_reference: `pingpong_controller/outputs/rl_sim/wmj_v1/training_seed_bank_a02/training_bank.json`; verification_reference: `pingpong_controller/outputs/rl_sim/wmj_v1/training_seed_bank_restore_a01/result.json` |
| `new_pose_model_audit` | complete_not_qualified_no_parameter_updates | 未定义/未定义 | 见CFG子字段 |
| `targeted_supplement` | complete_frozen_audit_unqualified_awaiting_next_QA | 未定义/未定义 | 见CFG子字段 |
| `auxiliary_refresh` | complete_mixed_validation_candidate_not_promoted | 未定义/未定义 | candidate_checkpoint: `pingpong_controller/outputs/rl_sim/wmj_v1/targeted_aux_refit_20261110_a01/auxiliary_u008192.json` |
| `bounded_imagination_pilot` | complete20_actor80_critic_mixed_native_outcomes_not_promoted | True/未定义 | candidate_checkpoint: `pingpong_controller/outputs/rl_sim/wmj_v1/targeted_aux_refit_20261110_a01/auxiliary_u008192.json` |
| `frozen_attribution_diagnostic` | complete_directional_sensitivity_awaiting_user_QA | 未定义/未定义 | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/attribution_plan_20261110_a02.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/frozen_attribution_verification_20261110_a01.json` |
| `module_gradient_diagnostic` | complete_numeric_hotspots_identified_awaiting_user_QA | 未定义/未定义 | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/module_gradient_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/module_gradient_verification_20261110_a01.json` |
| `mixed_precision_candidate` | frozen_validation_complete_partial_gradient_improvement_not_promoted | 未定义/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/mixed_precision_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/mixed_precision_verification_20261110_a01.json` |
| `paired_precision_trial` | paired_ten_update_trial_complete_not_promoted | True/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/paired_precision_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/paired_precision_verification_20261110_a01.json` |
| `paired_frozen_attribution` | frozen_attribution_recorded_collision_replay_limits | True/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/paired_attribution_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/paired_attribution_verification_20261110_a01.json` |
| `pose_margin_candidate` | frozen_calibrated_default_off | 未定义/未定义 | verification: `pingpong_controller/outputs/rl_sim/wmj_v1/pose_margin_verification_20261110_a01.json` |
| `pose_margin_trial` | ten_update_trial_evaluated_not_promoted | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/pose_trial_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/pose_trial_verification_20261110_a01.json` |
| `orientation_limit_trial` | frozen_comparison_complete_not_promoted | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/orientation_limit_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/orientation_limit_verification_20261110_a01.json` |
| `free_imagination_candidate` | frozen_audit_complete_with_reference_gaps | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/free_imagination_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/free_imagination_verification_20261110_a01.json` |
| `outgoing_velocity_reward_candidate` | frozen_calibration_complete_no_common_coefficient | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/velocity_reward_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/velocity_reward_verification_20261110_a01.json` |
| `expanded_error_attribution` | frozen_attribution_complete_auxiliary_gap_and_control_tail | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/expanded_error_attribution_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/expanded_error_attribution_verification_20261110_a01.json` |
| `expanded_auxiliary_refresh` | complete_paired_validation_not_promoted | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/expanded_auxiliary_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/expanded_auxiliary_verification_20261110_a01.json`; candidate_checkpoint: `pingpong_controller/outputs/rl_sim/wmj_v1/expanded_auxiliary_fit_20261110_a01/auxiliary_u012288.json` |
| `projection_consistency_candidate` | closed_independent_coordinate_precision_branch_repair_verified | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/projection_consistency_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/projection_consistency_verification_20261110_a01.json`; candidate_implementation: `world_model.projection_precision.make_runtime` |
| `corrected_candidate_audit` | closed_frozen_recheck_complete_not_qualified | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/corrected_candidate_plan_20261110_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/corrected_candidate_verification_20261110_a01.json`; candidate_checkpoint: `pingpong_controller/outputs/rl_sim/wmj_v1/expanded_auxiliary_fit_20261110_a01/auxiliary_u012288.json` |
| `corrected_reward_audit` | closed_frozen_recheck_complete_not_qualified | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/corrected_reward_plan_20261111_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/corrected_reward_verification_20261111_a01.json` |
| `reward_direction_audit` | closed_frozen_comparison_complete_not_qualified | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/reward_direction_plan_20261111_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/reward_direction_verification_20261111_a01.json` |
| `critic_target_audit` | closed_frozen_target_mask_audit_complete | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_target_plan_20261111_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_target_verification_20261111_a01.json` |
| `critic_only_fit` | closed_512_updates_complete_not_qualified | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_only_fit_plan_20261111_a02.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_only_fit_verification_20261111_a01.json`; candidate_checkpoint: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_only_fit_20261111_a01/learner_critic_added512.json` |
| `critic_ema_pair` | closed_paired_complete_relative_gain_not_qualified | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_ema_pair_plan_20261111_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_ema_pair_verification_20261111_a01.json` |
| `research_candidates_20261112` | 见对应配置 | 未定义/未定义 | 见CFG子字段 |
| `differentiability_requirement` | implemented_piecewise_path_gradient_tail_and_end_to_end_speedup_not_qualified | 未定义/未定义 | 见CFG子字段 |
| `critic_distribution_audit` | closed_frozen_complete_value_not_qualified | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_distribution_plan_20261112_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_distribution_verification_20261112_a01.json` |
| `critic_fixed_data_fit` | complete_fit_improved_heldout_not_qualified | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_fixed_fit_resume_plan_20261112_a01.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_fixed_fit_interruption_verification_20261112_a01.json`; final_verification: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_fixed_fit_verification_20261112_a02.json`; candidate_checkpoint: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_fixed_fit_resume_20261112_a01/critic_fixed_added512.json` |
| `critic_input_audit` | complete_frozen_input_audit_no_promotion | False/False | plan: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_input_audit_plan_20261112_a02.json`; verification: `pingpong_controller/outputs/rl_sim/wmj_v1/critic_input_audit_verification_20261112_a01.json` |

## F. 用户QA记录

以下为CFG已登记的问答，作为技术决策历史。本文主文档另外补齐截至a52的最新PSI、预算、完成状态与尚未回答的下一阶段问题。没有追加参数更新权限。

```json
[
  {
    "id": "fixed_namespace",
    "value": "WMJ-V1; no automatic major/physics/execution changes"
  },
  {
    "id": "domain",
    "value": "expanded candidate; Y upper remains -0.280m"
  },
  {
    "id": "attitude",
    "value": "roll/pitch +/-2deg"
  },
  {
    "id": "camera_noise",
    "value": "S01-S04 zero; S05 V324 1x AR1; supersedes initial-AR1 choice"
  },
  {
    "id": "view",
    "value": "expanded simulated XY view; not measured hardware coverage"
  },
  {
    "id": "goal_count",
    "value": "16 total; 10 consecutive stable starting no later than contact3; survive remainder"
  },
  {
    "id": "storage",
    "value": "6GiB total new outputs; at least8GiB free; stop and ask when insufficient"
  },
  {
    "id": "refresh_identity",
    "value": "checkpoint hash/model_revision/run_id may change; experiment contract stays fixed"
  },
  {
    "id": "transition_distance",
    "value": "XYZ 3D Euclidean distance"
  },
  {
    "id": "recovery_disturbance",
    "value": "S04/S05: once/sequence at actual-apex goal switch; external force20ms; |delta_vxy|<=0.05m/s, |delta_vz|<=0.10m/s; balanced directions; continuous physical/control/camera state"
  },
  {
    "id": "frozen_validation",
    "value": "128 cases/stage; nominal/near_side/mirror/boundary32 each; deterministic+sampled paired; max60s; goals1/2/2/4/4; every group/mode success>=90%; first physical failure fails case; unknown feasibility separate"
  },
  {
    "id": "history_window",
    "value": "640ms; explicit history plus structured physical world model; no VAE/latent predictor initially"
  },
  {
    "id": "history_content",
    "value": "only observed ball positions paired with racket position/orientation at each ball source timestamp; retain current q/dq/q_ref_error/accepted_action only"
  },
  {
    "id": "history_multirate_correction",
    "value": "ball-sample-indexed sparse racket history; no continuous200Hz racket stream in actor; high-rate feedback only for causal alignment"
  },
  {
    "id": "self_collision_admission",
    "value": "convex broad phase plus original triangle/containment refinement; ambiguous cases UNKNOWN; no extra clearance or mesh repair"
  },
  {
    "id": "initialization_audit_cost_scope",
    "value": "offline initialization range and seed selection only; no self-collision/geometry/IK scans in training steps or runtime resets; cached preaudited seeds; preparation wall time counted"
  },
  {
    "id": "numerical_comparison",
    "value": "User accepts recorded tiny CUDA differences; do not block on floating-point bit identity or repeat expensive validation for insignificant roundoff; retain physical/discrete checks",
    "date": "2026-11-06"
  },
  {
    "id": "history_representation_reconfirmation",
    "value": "Keep969D direct MLP; prefix73 plus temporal128 encoder and full latent world model remain candidates only, without automatic activation",
    "date": "2026-11-06"
  },
  {
    "id": "world_observation_roll_forward",
    "value": "Predict only new physical state; carry existing observed history forward with known camera, holds, timestamp alignment and age rules; never re-predict old rows or inject future MJX observations into imagination",
    "date": "2026-11-06"
  },
  {
    "id": "directed_conditions_and_zero_initial_spin",
    "value": "Incoming vz[-3.5,-1.0]m/s, horizontal norm<=0.5m/s, contact lead80-160ms; every new ball initialization has zero angular velocity; preserve and record natural collision-induced rotation",
    "date": "2026-11-06"
  },
  {
    "id": "high_fidelity_proxy_assumption",
    "value": "User considers existing simulation/contact/control highly similar to hardware based on prior FPLC experiments; reuse that baseline as the proxy real environment for MBRL and focus new fidelity checks on world-model versus MJX outcomes",
    "date": "2026-11-06"
  },
  {
    "id": "contact_input_dimension_reconfirmation",
    "value": "Keep approved39-D contact inputs and optimize fitting; decline proposed46-D velocity-tracking-error comparison; no added contact dq/ddq; retain existing47-D robot proxy",
    "date": "2026-11-09"
  },
  {
    "id": "contact_residual_off_integration",
    "value": "User selected full normal+tangent prior plus frozen auxiliary with39-D contact residual disabled; retain robot proxy and residual checkpoints; validate control/history/action effects; no Actor training authorized by this phase",
    "date": "2026-11-09"
  },
  {
    "id": "imagination_system_no_update_phase",
    "date": "2026-11-09",
    "value": "Implement reward, actor/critic, persistent state and save/restore; frozen world; validate gradients/continuity without actor or critic optimizer updates"
  },
  {
    "id": "independent_goal_reward_calibration",
    "date": "2026-11-09",
    "value": "User approves proposing explicit candidate coefficients and frozen trajectory scale/gradient calibration; no legacy fixed-anchor penalty and no actor/critic training"
  },
  {
    "id": "critic_complete_control_memory",
    "date": "2026-11-09",
    "value": "Structured physics/task/current actor observation plus complete RMP/filter/delay memory;256->256->1 with zero initial value; explicit fixed scales and no RNG/future labels;actor969 unchanged"
  },
  {
    "id": "course_dependent_branch_cold_start",
    "date": "2026-11-09",
    "value": "User approves course-dependent P_start counts and explicit temporary reassignment of missing enabled categories to precontact; causal complete imagined seeds fill categories later; no actor/critic updates"
  },
  {
    "id": "continuous_four_goal_training_sequences",
    "date": "2026-11-09",
    "value": "User approves continuous4-goal training groups with uniform one-of-three internal switch pulse; do not reset at group edges or force pulse after failure/time limit"
  },
  {
    "id": "critic_current_sequence_and_force_memory",
    "date": "2026-11-09",
    "value": "User approves6 current values,2602->256->256->1; retain969-D actor; no future pulse/goal/RNG inputs and no optimizer updates"
  },
  {
    "id": "frozen_case_spatial_distributions",
    "date": "2026-11-09",
    "value": "User approves full-domain stratified nominal, nearY[-.38,-.28], whole-sequenceX mirror with independent initialization audit, and2cm candidate-box face band;32 each;UNKNOWN/shortfall explicit;no policy filtering"
  },
  {
    "id": "frozen_standard_stationary_drop",
    "date": "2026-11-09",
    "value": "User approves warmed legal racket/controller near first goalXY, ball at first goalXYZ with zero initial linear/angular velocity;initial apex not scored;continuous subsequent state;no actor/critic updates"
  },
  {
    "id": "offline_warmed_flat_initialization_search",
    "date": "2026-11-09",
    "value": "User approves offline warmed-state initialization search, preferring flatter racket poses while preserving all fixed goals and constraints; no policy-score selection, no runtime geometry overhead, no Actor/Critic/world updates."
  },
  {
    "id": "independent_training_seed_bank_and_frozen_model_audit",
    "date": "2026-11-09",
    "source": "user_dialog_reply",
    "version_change": false,
    "value": "建立独立于固定128例验证集的全域训练种子库，核对完整历史/控制状态，并验证冻结世界模型在新姿态上的误差；Actor/Critic/world均不更新"
  },
  {
    "id": "independent_targeted_supplement_1024x128",
    "date": "2026-11-09",
    "value": "用户确认先做独立定向补采与冻结误差检查；最多1024环境x128控制步，两个64步块；新平拍、边缘及14缺口；不复用固定验证初态、不更新任何参数，V324和WMJ-V1及6GiB预算不变。"
  },
  {
    "id": "targeted_auxiliary_only_refresh",
    "date": "2026-11-10",
    "value": "User authorized solver-label replay of same supplement and only47->128->128->6 auxiliary fine-tune;at most4096 additional updates;half old/half new training;preserve normalization;paired old/new H1/8/32/64/128 validation;freeze all other models;no new scenarios;existing storage limits"
  },
  {
    "id": "bounded_S01_imagination_pilot",
    "date": "2026-11-10",
    "decision": "User approved candidate action-effect/gradient recheck followed conditionally by maximum20Actor and80Critic pure-imagination updates;native frozen paired evaluation every10updates;unchanged model/default/version/storage contracts"
  },
  {
    "topic": "frozen_pilot_attribution",
    "answer": "先冻结诊断，再决定训练或奖励调整",
    "scope": "baseline/u10/u20 and existing cases only;MJX/model outgoing/apex/reward/gradient and nominal_23 collision identity;<=8MiB;no updates or contract changes"
  },
  {
    "topic": "modulewise_frozen_gradient_diagnostic",
    "answer": "先分模块定位梯度不稳定",
    "scope": "existing independent training seeds;control/projection+FK,execution,soft contact+auxiliary;finite differences+local high precision;no network/reward/constraint/version changes;<=1MiB output"
  },
  {
    "topic": "mixed_precision_candidate_frozen_validation",
    "answer": "先做混合精度候选与冻结验证",
    "scope": "private candidate;H64gradients,savedMJXactioneffects,branch/runtime;no updates/promotion;<=2MiB"
  },
  {
    "topic": "paired_precision_trial",
    "answer": "开展配对10次试训",
    "scope": "two arms sameu20 Actor/Critic/Adam;no-update+save/restore first;10Actor40Critic each;fixed128det/sample evaluation;worldfrozen;no promotion"
  },
  {
    "topic": "paired_precision_trial_storage",
    "answer": "提高至6.125GiB，保留8GiB空闲",
    "phase_max_bytes": 134217728,
    "total_max_bytes": 6576668672
  },
  {
    "topic": "post_paired_trial_frozen_attribution",
    "answer": "先做上述冻结归因（推荐）",
    "scope": "u20/u30;existing independent training seeds;reward/gradient andmirror_15collision;no parameter/reward/constraint changes;2MiB phase;6.125GiB total;8GiB free"
  },
  {
    "topic": "continuous_pose_margin_frozen_calibration",
    "answer": "先校准连续姿态余量惩罚候选（推荐）",
    "scope": "default-off independent candidate;existing frozen sources;coefficient andheight/pose gradient calibration;no updates/default replacement;2MiB phase;6.125GiB total;8GiB free"
  },
  {
    "topic": "pose_margin_trial",
    "answer": "接入验收通过后试训10次（推荐）",
    "scope": "4/s independent reward;FP32u20 same learning/pool;no-update andrestore first;max10Actor40Critic;fixed128native;world frozen;no default promotion"
  },
  {
    "topic": "pose_margin_trial_storage",
    "answer": "阶段48 MiB、总额增加16 MiB（推荐）",
    "scope": "total6.140625GiB;freefloor8GiB;actual budget failures stop andask"
  },
  {
    "topic": "remove_racket_roll_pitch_limit_frozen_native_comparison",
    "answer": "我觉得可以限制再大一点，比如20度或者干脆取消限制。我觉得取消限制",
    "scope": "candidate removesRPprojectionandtermination;original2deg frozencontrol;sameFP32u30and128cases;zero updates;8MiBphase;6.140625GiBtotal;8GiBfree"
  }
]
```

## G. WM模块索引

模块docstring概括其原始职责，可能保留早期“independent/not integrated”等文字；当前组合以调用链和最新plan为准。顶层符号索引用于快速定位，不是替代源码。

| 模块（相对WM） | 职责 | 顶层类/函数 |
| --- | --- | --- | --- |
| [__init__.py](../rl_sim/world_model/__init__.py) | Independent, explicitly selected WMJ-V1 simulation research modules. |  |
| [action_effects.py](../rl_sim/world_model/action_effects.py) | Offline action-effect labels and paired changes; missing outcomes stay visible. | `model_event_trace`, `summarize_outcomes`, `compare_effects`, `compare_with_repeat` |
| [actor_critic.py](../rl_sim/world_model/actor_critic.py) | Pure imagined actor/critic objectives with explicit termination semantics. | `ImaginedBatch`, `validate_batch`, `lambda_returns`, `actor_return`, `frozen_value`, `init_critic`, `critic_loss`, `average_gradients`, `target_ema`, `apply_accumulated_update` |
| [artifacts.py](../rl_sim/world_model/artifacts.py) | Unique, integrity-checked evidence writes inside the shared storage cap. | `discard_file_cache`, `write_json_unique`, `write_array_archive_unique`, `read_array_archive` |
| [attribution_checks.py](../rl_sim/world_model/attribution_checks.py) | Attribution-only physical input comparison, independent of solver scratch. | `world_event_trace`, `compare_initial_interfaces` |
| [attribution_common.py](../rl_sim/world_model/attribution_common.py) | Shared immutable population and budgets for frozen pilot attribution only. | `apex_replay_deltas`, `plan_and_budget`, `worker_check`, `source_report`, `population`, `actors`, `initial_interface`, `frozen_models` |
| [audit_action_effects.py](../rl_sim/world_model/audit_action_effects.py) | Same-full-state MJX/model action pulses with frozen model and raw continuation. | `clone_batch_lanes`, `initial_world`, `audit`, `main` |
| [audit_attribution_collision.py](../rl_sim/world_model/audit_attribution_collision.py) | Original evaluator graph, host-only collision inspection; no new trace outputs. | `original_policy_step`, `audit`, `main` |
| [audit_attribution_modes.py](../rl_sim/world_model/audit_attribution_modes.py) | Frozen own-policy world probes with actual1ms contact-mode comparisons. | `build_mode_step`, `audit`, `main` |
| [audit_attribution_native.py](../rl_sim/world_model/audit_attribution_native.py) | Replay baseline/u10/u20 without updates; identify native contact and rewards. | `contact_observer`, `audit`, `main` |
| [audit_attribution_native_gradients.py](../rl_sim/world_model/audit_attribution_native_gradients.py) | Physical finite differences along frozen world-return bias directions. | `native_reward_step`, `audit`, `main` |
| [audit_attribution_world.py](../rl_sim/world_model/audit_attribution_world.py) | Frozen native-action world replay and policy-return derivative attribution. | `target_critic`, `audit`, `main` |
| [audit_auxiliary_refresh.py](../rl_sim/world_model/audit_auxiliary_refresh.py) | Paired frozen parent/candidate own-state rollouts on all old/new validation. | `audit`, `main` |
| [audit_collision_geometry.py](../rl_sim/world_model/audit_collision_geometry.py) | Reproduce a convex-mesh alarm from an authenticated native reset artifact. | `mesh_world`, `_ancestry`, `_save_bytes`, `mesh_components`, `inspect_pair`, `main` |
| [audit_contact_fit.py](../rl_sim/world_model/audit_contact_fit.py) | Separate one-step contact fit error from published recursive validation. | `main` |
| [audit_contact_kernel.py](../rl_sim/world_model/audit_contact_kernel.py) | Compare actual V324 sphere-mesh kernel, Euclidean prism and saved labels. | `main` |
| [audit_contact_solver.py](../rl_sim/world_model/audit_contact_solver.py) | One native acquisition with read-only solver rows; no model fitting. | `solver_rows`, `audit`, `main` |
| [audit_control_components.py](../rl_sim/world_model/audit_control_components.py) | Recorded-reference diagnostics separating projection from RMP arithmetic. | `make_component_replays` |
| [audit_corrected_candidate.py](../rl_sim/world_model/audit_corrected_candidate.py) | Frozen u8192/u12288 model and action checks under corrected projection. | `audit`, `main` |
| [audit_corrected_reward.py](../rl_sim/world_model/audit_corrected_reward.py) | Frozen u12288 inverse-speed reward scale and H64 derivative recheck. | `run_mode`, `audit`, `main` |
| [audit_course_pool.py](../rl_sim/world_model/audit_course_pool.py) | Guarded 1024-lane course/pool/gradient validation, with zero updates. | `select_initial_seeds`, `summarize_windows`, `audit`, `main` |
| [audit_critic_distribution.py](../rl_sim/world_model/audit_critic_distribution.py) | Frozen train/heldout source comparison; no optimizer or model updates. | `check_plan`, `window_values`, `run`, `main` |
| [audit_critic_inputs.py](../rl_sim/world_model/audit_critic_inputs.py) | Read-only four-family Critic input and generalization audit on fixed data. | `stats`, `check_plan`, `gradient_summary`, `derivative_check`, `coverage_report`, `run`, `main` |
| [audit_critic_targets.py](../rl_sim/world_model/audit_critic_targets.py) | Frozen target/mask audit on the registered32sources and existing critics. | `audit_window`, `audit`, `main` |
| [audit_error_attribution.py](../rl_sim/world_model/audit_error_attribution.py) | Frozen attribution of expanded-attitude control, robot and contact errors. | `scopes`, `command_errors`, `control_diagnostics`, `local_diagnostics`, `model_windows`, `audit`, `main` |
| [audit_expanded_retention.py](../rl_sim/world_model/audit_expanded_retention.py) | Reuse all original validation cases without duplicating per-step arrays. | `retention_source`, `audit` |
| [audit_fk_context.py](../rl_sim/world_model/audit_fk_context.py) | Compare seven-joint FK's frozen support context against actual teacher poses. | `main` |
| [audit_free_imagination.py](../rl_sim/world_model/audit_free_imagination.py) | Frozen free-attitude control and model audit on independent training sources. | `take_native`, `restore_selected`, `restore_native_sources`, `pose_angles`, `maximum_tree_delta`, `endpoint_bounds`, `audit`, `main` |
| [audit_full_prior_world.py](../rl_sim/world_model/audit_full_prior_world.py) | Guarded held-out integration audit: own controls versus recorded commands. | `AuditReader`, `read_selected_trial`, `make_bank_builder`, `cached_pose_sequence`, `teacher_seeds`, `projection_diagnostic`, `audit`, `release_cpu_heap`, `main` |
| [audit_gpu_replay.py](../rl_sim/world_model/audit_gpu_replay.py) | Guarded small CUDA replay audit; never a trainer or a collection qualification. | `_gpu_sample`, `main` |
| [audit_mixed_actions.py](../rl_sim/world_model/audit_mixed_actions.py) | All480 existing saved MJX forks against the frozen mixed numerical candidate. | `pack_rows`, `pack_numeric_slabs`, `audit_actions` |
| [audit_mixed_precision.py](../rl_sim/world_model/audit_mixed_precision.py) | Bounded frozen H64 probes for explicit WMJ precision runtimes. | `phase_plan`, `check`, `rollout_function`, `difference_row`, `audit`, `main` |
| [audit_module_gradient_details.py](../rl_sim/world_model/audit_module_gradient_details.py) | Private follow-up: paired projection branches and auxiliary input rounding. | `compare_precisions`, `input_rounding_rows`, `run_details` |
| [audit_module_gradients.py](../rl_sim/world_model/audit_module_gradients.py) | Frozen module derivatives on independent training seeds;1MiB output scope. | `phase_plan`, `checkpoint`, `load_runtime`, `capture_states`, `geometry_modes`, `normalized_directions`, `component_functions`, `audit`, `main` |
| [audit_native.py](../rl_sim/world_model/audit_native.py) | Bounded CPU audit of the actual WMJ/V324 constructor, camera and full seed. | `compare_camera`, `main` |
| [audit_native_contact_geometry.py](../rl_sim/world_model/audit_native_contact_geometry.py) | Check contact geometry/gating against real replayed solver intervals. | `main` |
| [audit_new_pose_world.py](../rl_sim/world_model/audit_new_pose_world.py) | Frozen model audit on independently cached flat validation poses; no updates. | `native_world`, `finite_intervals`, `archive_finite`, `audit`, `main` |
| [audit_normal_components.py](../rl_sim/world_model/audit_normal_components.py) | Paired H8 error decomposition using the exact existing validation windows. | `component_errors`, `audit`, `main` |
| [audit_orientation_initial.py](../rl_sim/world_model/audit_orientation_initial.py) | Preparation-only native pairing diagnostic; no policy step or parameter update. | `Prepared`, `main` |
| [audit_paired_attribution.py](../rl_sim/world_model/audit_paired_attribution.py) | Frozen full-Actor reward attribution and original-graph collision inspection. | `plan_and_budget`, `check`, `policies`, `shifted`, `gradients`, `collisions`, `main` |
| [audit_pilot_actions.py](../rl_sim/world_model/audit_pilot_actions.py) | Frozen auxiliary candidate against authenticated, saved native action forks. | `audit`, `main` |
| [audit_pilot_gradients.py](../rl_sim/world_model/audit_pilot_gradients.py) | Candidate-only full-batch gradient admission; no optimizer mutation. | `audit`, `main` |
| [audit_projection_consistency.py](../rl_sim/world_model/audit_projection_consistency.py) | Frozen native-versus-imagination projection branch diagnosis, no updates. | `summarize`, `audit`, `main` |
| [audit_projection_precision.py](../rl_sim/world_model/audit_projection_precision.py) | Replay saved projection inputs with local FP32 coordinate transforms only. | `native_methods`, `delta`, `audit`, `main` |
| [audit_projection_precision_integrated.py](../rl_sim/world_model/audit_projection_precision_integrated.py) | Frozen full native/controller pairing of the local coordinate precision fix. | `install_candidate`, `audit`, `main` |
| [audit_projection_seed.py](../rl_sim/world_model/audit_projection_seed.py) | One-step native-MJX versus world projection from the same full1024-lane seed. | `projection_inputs`, `highest_matrix_fk`, `audit`, `main` |
| [audit_projection_variants.py](../rl_sim/world_model/audit_projection_variants.py) | Compare private FK implementations on the fixed integration control population. | `audit`, `main` |
| [audit_reward_direction.py](../rl_sim/world_model/audit_reward_direction.py) | Frozen candidate-gradient directions evaluated in native MJX and imagination. | `make_noise`, `audit`, `main` |
| [audit_saved_pose_events.py](../rl_sim/world_model/audit_saved_pose_events.py) | Replay saved new-pose model branches for event-aligned outcome diagnostics. | `audit`, `main` |
| [audit_soft_acceleration.py](../rl_sim/world_model/audit_soft_acceleration.py) | Offline oracle: how much prior error is due to ignored racket acceleration? | `main` |
| [audit_support_replay.py](../rl_sim/world_model/audit_support_replay.py) | Compare full-chain measured-state FK directly to native replay pose/twist. | `load_support_chain`, `audit`, `main` |
| [audit_tangent_components.py](../rl_sim/world_model/audit_tangent_components.py) | Paired H8 error decomposition using the exact existing validation windows. | `component_errors`, `audit`, `main` |
| [audit_tangent_prior.py](../rl_sim/world_model/audit_tangent_prior.py) | One-step normal/full prior diagnostics for coherent native replay states. | `prior_errors` |
| [audit_tangent_replay.py](../rl_sim/world_model/audit_tangent_replay.py) | Same-replay held-out normal/full prior diagnostic; no cross-replay labels. | `diagnostic_raw`, `audit`, `main` |
| [audit_targeted_world.py](../rl_sim/world_model/audit_targeted_world.py) | Frozen own-state model errors on the independent direct-support supplement. | `audit`, `main` |
| [audit_world_control.py](../rl_sim/world_model/audit_world_control.py) | CPU replay of full seeded controls against coherent recorded V324 commands. | `audit`, `main` |
| [audit_world_control_gpu.py](../rl_sim/world_model/audit_world_control_gpu.py) | Guarded GPU counterpart of the full-seed known-control replay audit. | `main` |
| [auxiliary_refresh.py](../rl_sim/world_model/auxiliary_refresh.py) | Explicit auxiliary-only refresh with inherited scales and optimizer memory. | `family_ids`, `validation_batches`, `refresh_indexes`, `restore_auxiliary` |
| [ball_flight.py](../rl_sim/world_model/ball_flight.py) | Known ball flight, separated from learned contact and task proposals. | `free_flight_substep`, `launch_to_contact`, `ballistic_apex`, `velocity_for_apex` |
| [branch_bank.py](../rl_sim/world_model/branch_bank.py) | Bounded CPU full-state seed cache, populated by actual causal events only. | `BranchSeedBank`, `prepare_training_window` |
| [build_training_seed_bank.py](../rl_sim/world_model/build_training_seed_bank.py) | Reconstruct and freeze a diverse training bank without parameter updates. | `bank_signature`, `first_goal`, `reset_source_template`, `load_training_bank`, `verify_saved_bank`, `build`, `main` |
| [calibrate_pose_margin.py](../rl_sim/world_model/calibrate_pose_margin.py) | Frozen pose-margin coefficient calibration; no optimizer or policy update. | `load_plan`, `check`, `audit`, `main` |
| [calibrate_velocity_reward.py](../rl_sim/world_model/calibrate_velocity_reward.py) | Frozen replacement velocity-cost calibration; no optimizer or parameter update. | `objective_function`, `coefficient_grid`, `local_native_calibration`, `audit`, `main` |
| [collect_refresh_labels.py](../rl_sim/world_model/collect_refresh_labels.py) | Compact solver labels on the same authorized targeted acquisition. | `collect`, `main` |
| [collect_solver_auxiliary.py](../rl_sim/world_model/collect_solver_auxiliary.py) | Read-only replay labels for the authorized independent 47->6 diagnostic. | `solver_labels`, `collect`, `main` |
| [collect_targeted.py](../rl_sim/world_model/collect_targeted.py) | One independent1024x128 full-support collection with frozen teacher weights. | `collect`, `main` |
| [collect_teacher.py](../rl_sim/world_model/collect_teacher.py) | Collect warmed full-domain V324 experiments without updating any policy. | `compact_state`, `_stack_cpu`, `save_seed`, `run_collection`, `main` |
| [collection_labels.py](../rl_sim/world_model/collection_labels.py) | CPU labels separating task bounds from usable teacher dynamics samples. | `label_task_bounds` |
| [contact_checkpoint.py](../rl_sim/world_model/contact_checkpoint.py) | Narrow source compatibility for the unchanged legacy contact calculation. | `legacy_geometry_dispatch_compatible` |
| [contact_data.py](../rl_sim/world_model/contact_data.py) | Bounded real contact windows, event-balanced sampling and physical labels. | `check_continuity`, `read_contact_trial`, `contact_window_rows`, `take_contact_windows`, `prepare_contact_batch` |
| [contact_dynamics.py](../rl_sim/world_model/contact_dynamics.py) | Finite-duration contact plus known 1 ms flight, with persistent contact age. | `ContactState`, `contact_step`, `contact_rollout` |
| [contact_fitting.py](../rl_sim/world_model/contact_fitting.py) | Optional single-step supervision alongside unchanged recursive contact loss. | `make_contact_objective`, `validate_contact_resume_recipe` |
| [contact_geometry.py](../rl_sim/world_model/contact_geometry.py) | Select legacy prism or the teacher's actual MJX sphere/mesh geometry. | `MJXContactSurface`, `surface_from_compiled_model`, `surface_features` |
| [contact_losses.py](../rl_sim/world_model/contact_losses.py) | Physical-scale contact state, edge and outgoing-outcome supervision. | `masked_mean`, `balanced_edge_loss`, `ballistic_apex`, `contact_module_loss` |
| [contact_model.py](../rl_sim/world_model/contact_model.py) | Finite-duration 1 ms contact residual in a physically consistent local frame. | `world_spin`, `contact_features`, `ContactPrediction`, `contact_prediction` |
| [contact_rates.py](../rl_sim/world_model/contact_rates.py) | Explicit future learning-rate stages with preserved optimizer history. | `parse_rates`, `rate_at`, `rate_schedule`, `validate_rate_resume` |
| [contact_regularization.py](../rl_sim/world_model/contact_regularization.py) | Training-only activation penalty for the unchanged39-D tanh contact model. | `activation_penalty`, `regularized_contact_objective`, `validate_activation_transition` |
| [contact_schedule.py](../rl_sim/world_model/contact_schedule.py) | Contact fitting window schedules and exact consumed-data accounting. | `contact_schedule`, `validate_contact_training_recipe`, `contact_substep_uses` |
| [contact_surface.py](../rl_sim/world_model/contact_surface.py) | Finite compiled racket-prism features for experiments and learned contact. | `RacketSurface`, `surface_from_vertices`, `surface_features`, `radial_boundary_distance`, `proposed_ball_center` |
| [contracts.py](../rl_sim/world_model/contracts.py) | Configuration identities and authenticated inputs for fixed WMJ-V1. | `repository_root`, `canonical_json`, `config_hash`, `file_hash`, `authenticate`, `_require`, `validate_config`, `load_config` |
| [control_execution.py](../rl_sim/world_model/control_execution.py) | Private reuse of the authenticated V324 EXEC projection-loop transformation. | `_projection`, `projection_v324` |
| [control_kinematics.py](../rl_sim/world_model/control_kinematics.py) | Private native-quaternion ancestry FK for projection implementation audits. | `ControlChain`, `extract_control_chain`, `forward_control_kinematics` |
| [control_projection.py](../rl_sim/world_model/control_projection.py) | V324 bounded reference projection with full-chain differentiable FK. | `ProjectionContract`, `ProjectionResult`, `project_reference` |
| [control_rmp.py](../rl_sim/world_model/control_rmp.py) | Native RMP with per-lane ring clocks and separately preserved output delay. | `promote_rmp_indices`, `rmp_step_lanes`, `apply_output_delay` |
| [control_seed.py](../rl_sim/world_model/control_seed.py) | Restore authentic controller memory without loading an entire MJX seed. | `read_control_seed` |
| [corrected_action_audit.py](../rl_sim/world_model/corrected_action_audit.py) | Same-batch native action forks from the32 corrected-control reset sources. | `first_failure`, `run_actions` |
| [corrected_candidate.py](../rl_sim/world_model/corrected_candidate.py) | Frozen corrected-control captures and explicitly paired action pulses. | `capture_source`, `make_capture`, `pulse_assignments`, `branch_actions`, `resolved_effects` |
| [corrected_reward.py](../rl_sim/world_model/corrected_reward.py) | Diagnostic-only reward returns and directional checks under corrected control. | `objective_returns`, `objective_function`, `directional_summary`, `native_reward_calibration` |
| [course_control.py](../rl_sim/world_model/course_control.py) | WMJ-only derivative adapter, with exactly the native RMP forward arithmetic. | `braking_sqrt`, `_braking_sqrt_jvp`, `_verify_sources`, `joint_rmp_step`, `rmp_step_lanes`, `control_step` |
| [course_env.py](../rl_sim/world_model/course_env.py) | Opt-in course environment with explicit current pulse/sequence memory. | `init_course`, `course_critic_features`, `course_critic_layout`, `course_step` |
| [course_goals.py](../rl_sim/world_model/course_goals.py) | Finite cached goal catalog and conditional50/30/20 task sampling. | `GoalTable`, `build_goal_table`, `allowed_goals`, `validate_start_goals`, `draw_next_goals` |
| [course_learner.py](../rl_sim/world_model/course_learner.py) | No-update course integration with explicit preflight and commit boundaries. | `CompactCourseWindow`, `compact_course_window`, `compact_critic_arrays`, `validate_course_config`, `make_course_gradient`, `course_gradient_window`, `commit_course_windows`, `prepare_course_session`, `course_runtime_signature` |
| [course_rollout.py](../rl_sim/world_model/course_rollout.py) | Causal course rollouts and bounded complete-state event capture. | `CapturedSeed`, `_empty_capture`, `_capture`, `course_policy_window`, `course_actor_objective`, `record_captured_seeds`, `validate_course_window` |
| [course_sampling.py](../rl_sim/world_model/course_sampling.py) | Approved course-dependent P_start with explicit cold-start reassignment. | `sample_course_starts` |
| [course_schedule.py](../rl_sim/world_model/course_schedule.py) | Causal four-goal groups and20 ms external-force scheduling at actual apexes. | `ScheduleState`, `init_schedule`, `pulse_proposal`, `finish_substep`, `schedule_features` |
| [critic_distribution.py](../rl_sim/world_model/critic_distribution.py) | Read-only value diagnostics by source, contact phase and failure distance. | `failure_horizon`, `contact_phase`, `compact_regression`, `stratified_summary` |
| [critic_features.py](../rl_sim/world_model/critic_features.py) | Explicit privileged critic layout with complete native controller memory. | `feature_blocks`, `critic_features`, `critic_layout` |
| [critic_fit.py](../rl_sim/world_model/critic_fit.py) | Opt-in Critic-only transactions, exact parent learning state and seed resets. | `restore_learning`, `make_update`, `update_session`, `prepare_fit_state`, `unchanged_actor_digest` |
| [critic_input_audit.py](../rl_sim/world_model/critic_input_audit.py) | Frozen feature scale, phase-matched coverage and value-input derivatives. | `sequential_storage_reservation`, `input_groups`, `component_stats`, `distribution_summary`, `input_gradients`, `_nearest_batch`, `nearest_training_rows` |
| [critic_targets.py](../rl_sim/world_model/critic_targets.py) | Read-only critic diagnostics with independent finite-return enumeration. | `lambda_mixture_oracle`, `stats`, `regression_stats`, `checkpoint_critics_and_pool`, `feature_summary`, `activation_summary`, `historical_learning` |
| [datasets.py](../rl_sim/world_model/datasets.py) | Stable group partitions and strict imagined-transition provenance. | `SplitRegistry`, `require_imagination` |
| [domain_atlas.py](../rl_sim/world_model/domain_atlas.py) | Policy-independent compiled geometry, bounded IK and velocity diagnostics. | `collision_pairs`, `check_self_collision`, `_rpy_and_rate`, `solve_ik`, `pure_vertical_speed_capacity`, `main` |
| [error_attribution.py](../rl_sim/world_model/error_attribution.py) | Read-only native-input substitutions for frozen error attribution only. | `capture_source`, `augmented_recorded_loop`, `instrument_with_solver`, `make_capture`, `window_arrays`, `largest_prefix_errors`, `native_contact_step`, `conditional_step` |
| [evaluate_frozen.py](../rl_sim/world_model/evaluate_frozen.py) | Guarded native fixed-case evaluation; no optimizer or learned-model updates. | `json_numeric`, `endpoint_code`, `select_evaluation_actor`, `audit`, `main` |
| [evaluate_orientation_limit.py](../rl_sim/world_model/evaluate_orientation_limit.py) | Fixed native128 deterministic/sampled evaluation of a bounded/free-racket-attitude frozen comparison. | `main` |
| [evaluate_paired.py](../rl_sim/world_model/evaluate_paired.py) | Fixed native128 deterministic/sampled evaluation of a paired u30 export. | `main` |
| [evaluate_physics.py](../rl_sim/world_model/evaluate_physics.py) | Paired held-out native-robot versus own-robot contact prediction. | `_summaries`, `evaluate`, `main` |
| [evaluate_pilot.py](../rl_sim/world_model/evaluate_pilot.py) | Evaluate an explicitly exported pilot actor using the unchanged native cases. | `main` |
| [evaluate_pose_trial.py](../rl_sim/world_model/evaluate_pose_trial.py) | Fixed native128 deterministic/sampled evaluation of a pose-margin trial u30 export. | `main` |
| [expanded_auxiliary.py](../rl_sim/world_model/expanded_auxiliary.py) | Bounded expanded-motion sources, labels and auxiliary checkpoint lineage. | `select_sources`, `retained_label_mask`, `excluded_family_mask`, `instrument_full_labels`, `source_capture_code`, `make_source_capture`, `restore_candidate` |
| [expanded_model_audit.py](../rl_sim/world_model/expanded_model_audit.py) | Paired free-attitude own-state predictions on explicitly retained sources. | `paired_windows` |
| [external_force.py](../rl_sim/world_model/external_force.py) | Known ball-force integration on the frozen residual-free physical model. | `forced_physics_step` |
| [fit_auxiliary_refresh.py](../rl_sim/world_model/fit_auxiliary_refresh.py) | Bounded auxiliary-only continuation on equal old/new training populations. | `fit`, `main` |
| [fit_contact.py](../rl_sim/world_model/fit_contact.py) | Fit a finite-duration contact module on actual MJX trajectories. | `validate_contact`, `fit`, `main` |
| [fit_expanded_auxiliary.py](../rl_sim/world_model/fit_expanded_auxiliary.py) | Auxiliary-only continuation with the approved 50/50 old/new source mixture. | `read_old_labels`, `fit` |
| [fit_physics_contact.py](../rl_sim/world_model/fit_physics_contact.py) | Bounded39-D contact refit on frozen execution-proxy trajectories. | `fit`, `main` |
| [fit_robot.py](../rl_sim/world_model/fit_robot.py) | Fit and validate the robot execution component on immutable MJX data. | `dataset_identity`, `load_chain`, `summarize_errors`, `validate_robot`, `fit`, `main` |
| [fit_schedule.py](../rl_sim/world_model/fit_schedule.py) | Explicit 8/16/32/64 control-window curriculum with compatible resume. | `parse_schedule`, `controls_at`, `validate_resume_recipe` |
| [fit_solver_auxiliary.py](../rl_sim/world_model/fit_solver_auxiliary.py) | Bounded supervised diagnostic; never installs a component in the world. | `load_labels`, `family_pool`, `sample_pool`, `summarize_prediction`, `evaluate`, `fit`, `main` |
| [fit_support.py](../rl_sim/world_model/fit_support.py) | Fit the approved full-chain execution proxy using coherent native replay. | `join_trial`, `validate`, `fit`, `main` |
| [fixed_critic_fit.py](../rl_sim/world_model/fixed_critic_fit.py) | Critic-only transactions over immutable, parent-generated lambda labels. | `invariant_digest`, `update_fixed_state`, `next_draw` |
| [flat_initialization.py](../rl_sim/world_model/flat_initialization.py) | Offline horizontal-pose search and complete native-state lane selection. | `solve_flat_pose`, `pose_preference`, `merge_complete_native`, `candidate_targets`, `search_case`, `recover_nonexact_case` |
| [free_imagination.py](../rl_sim/world_model/free_imagination.py) | Explicit free-attitude imagination candidate; all shared paths stay intact. | `make_projector`, `FreeOrientationRuntime`, `make_runtime`, `select_verified_state` |
| [free_orientation.py](../rl_sim/world_model/free_orientation.py) | Private native-only ablation of racket roll/pitch constraints, default off. | `position_only_projection`, `remove_reference_attitude_backstops`, `install_position_only_reference`, `free_physical_bounds`, `phase_plan`, `audit` |
| [frozen_cases.py](../rl_sim/world_model/frozen_cases.py) | Policy-independent fixed128-case course manifests; no admission filtering. | `_stratified`, `_next_goal`, `generate_cases`, `validate_case_manifest` |
| [frozen_initialization.py](../rl_sim/world_model/frozen_initialization.py) | Offline standard-drop admission; never called from a training/evaluation tick. | `vertical_drop_geometry`, `StandardDropAuditor`, `main` |
| [frozen_seed.py](../rl_sim/world_model/frozen_seed.py) | Immutable complete warmed native populations shared by frozen evaluations. | `save_warmed_seed`, `load_warmed_seed` |
| [frozen_task.py](../rl_sim/world_model/frozen_task.py) | Finite, paired evaluation accounting over real substep event labels. | `EvaluationState`, `_select`, `init_evaluation`, `fail_evaluation`, `consume_events` |
| [full_prior_seeds.py](../rl_sim/world_model/full_prior_seeds.py) | Reconstruct complete full-chain teacher seeds from a coherent replay. | `read_initial_camera`, `initial_full_history`, `reconstruct_teacher_bank`, `select_teacher_seeds` |
| [full_prior_world.py](../rl_sim/world_model/full_prior_world.py) | Opt-in own-feedback world with frozen execution/auxiliary and no residual. | `FullPriorWorldState`, `measured_racket_pose`, `full_prior_physics_step`, `world_actor_observation`, `full_prior_world_step`, `full_prior_world_rollout` |
| [generators.py](../rl_sim/world_model/generators.py) | Full-domain teacher experiment proposals; outcomes always come from MJX. | `PoseCatalog`, `load_pose_catalog`, `select_pose_per_region`, `DirectedContactPlan`, `sample_contact_lead_controls`, `propose_directed_contacts`, `generator_assignments`, `correlated_action_sequences` |
| [geometry.py](../rl_sim/world_model/geometry.py) | Extract contact geometry from the instantiated model and audit native FK. | `audit_differentiable_fk`, `compiled_geometry` |
| [hybrid_contact.py](../rl_sim/world_model/hybrid_contact.py) | Optional soft-contact prior plus the unchanged learned 6-D correction. | `hybrid_contact_step`, `hybrid_contact_rollout`, `hybrid_contact_loss` |
| [imagination_checkpoint.py](../rl_sim/world_model/imagination_checkpoint.py) | Unique full numerical snapshots, authenticated before exact learner resume. | `runtime_signature`, `save_checkpoint`, `restore_checkpoint` |
| [imagination_diagnostics.py](../rl_sim/world_model/imagination_diagnostics.py) | Frozen continuation and piecewise gradient checks, without optimization. | `check_continuation_and_direction` |
| [imagination_learner.py](../rl_sim/world_model/imagination_learner.py) | Transactional learning primitives; this implementation phase is read-only. | `LearnerState`, `init_learner`, `slice_environment`, `make_actor_gradient`, `actor_gradient_window`, `commit_policy_windows`, `critic_training_arrays`, `critic_gradient_batch`, `apply_training_transaction` |
| [imagined_env.py](../rl_sim/world_model/imagined_env.py) | Pure imagined physical/control/history/task transitions with explicit endings. | `EpisodeContract`, `ImaginedState`, `ImaginedTransition`, `init_imagined`, `select_lanes`, `reset_lanes`, `task_substep`, `finite_physics`, `finite_lanes`, `physical_bounds`, `imagined_step` |
| [imagined_rewards.py](../rl_sim/world_model/imagined_rewards.py) | Independent goal shaping; candidate weights do not redefine task success. | `RewardContract`, `release_reward`, `control_reward`, `approach_reward` |
| [imagined_rollout.py](../rl_sim/world_model/imagined_rollout.py) | Reparameterized actor rollouts through a frozen structured world model. | `PolicyWindow`, `policy_window`, `window_batch`, `actor_window_objective` |
| [initialization_geometry.py](../rl_sim/world_model/initialization_geometry.py) | Offline-only seed geometry admission; never imported by a rollout step. | `MeshTopology`, `_separator`, `_winding_number`, `refine_mesh_pair`, `InitializationGeometryAuditor` |
| [kinematics.py](../rl_sim/world_model/kinematics.py) | Differentiable hinge-chain FK extracted from the instantiated MuJoCo model. | `Hinge`, `ChainBody`, `KinematicChain`, `Kinematics`, `_rotation`, `extract_chain`, `_axis_rotation`, `forward_kinematics` |
| [mesh_surface.py](../rl_sim/world_model/mesh_surface.py) | Bounded original-triangle diagnostics, independent of convex hull collision. | `triangle_pairs_overlap`, `_Node`, `_tree`, `audit_surface_intersections` |
| [migration.py](../rl_sim/world_model/migration.py) | Authenticated V321 actor migration without loading its legacy simulator. | `_ActorArchiveReader`, `load_v321_actor`, `_validate_actor`, `expand_actor_73`, `expand_actor_history`, `actor_mean` |
| [mixed_action_seed.py](../rl_sim/world_model/mixed_action_seed.py) | Restore exact saved MJX action-fork inputs without compiling a new MJX reset. | `read_action_seed` |
| [mixed_precision.py](../rl_sim/world_model/mixed_precision.py) | Explicit, default-off WMJ numerical candidate; native FP32 control is retained. | `normalized_prediction`, `robot_increment_step`, `control_boundary`, `stable_course_counters`, `PrecisionRuntime`, `make_runtime` |
| [model_data.py](../rl_sim/world_model/model_data.py) | Bounded CPU batches for supervised dynamics modules, with stable splits. | `RobotChunkReader`, `robot_chunk`, `robot_statistics_update`, `sample_robot_windows` |
| [model_losses.py](../rl_sim/world_model/model_losses.py) | Normalized robot execution losses; composed event/outcome losses follow separately. | `racket_outputs`, `robot_execution_loss` |
| [model_numerics.py](../rl_sim/world_model/model_numerics.py) | Explicit learned-component precision, separate from native control arithmetic. | `precision_scope`, `validate_precision_transition` |
| [module_gradient_tools.py](../rl_sim/world_model/module_gradient_tools.py) | Diagnostic-only precision copies and directional checks; no production edits. | `private_clone`, `cast_floats`, `distribution`, `probe_direction` |
| [native_contact_data.py](../rl_sim/world_model/native_contact_data.py) | Contact windows conditioned on true native pose/twist from support replays. | `read_native_contact_trial`, `prepare_native_contact_batch` |
| [native_entry_collision.py](../rl_sim/world_model/native_entry_collision.py) | Host observation around the complete preserved native evaluation entry point. | `write_compact_json`, `observed_jit`, `audit` |
| [native_evaluation.py](../rl_sim/world_model/native_evaluation.py) | Private V324 task hook: observe native contacts and apply approved forces. | `_EvaluationHook`, `augment_source`, `native_event_trace`, `merge_observer_trace`, `evaluation_loop`, `build_evaluation_step` |
| [native_observation.py](../rl_sim/world_model/native_observation.py) | 969-D observational adapter around the untouched V324 teacher transition. | `racket_pose_from_chain`, `camera_from_native`, `NativeObservationAdapter` |
| [native_pair.py](../rl_sim/world_model/native_pair.py) | Task-scaled pairing of the private native evaluation graph. | `assess_native_pair` |
| [native_stack.py](../rl_sim/world_model/native_stack.py) | Private WMJ adapters over the authenticated V324 simulator, never shared edits. | `load_native_stack`, `build_native_environment` |
| [neural.py](../rl_sim/world_model/neural.py) | Small FP32 JAX networks and serializable Adam without optional dependencies. | `init_mlp`, `mlp`, `AdamState`, `init_adam`, `adam_update` |
| [new_pose_audit.py](../rl_sim/world_model/new_pose_audit.py) | Offline window and validity rules for frozen new-pose model diagnostics. | `prefix_validity`, `diagnostic_starts`, `select_windows` |
| [normal_checkpoint.py](../rl_sim/world_model/normal_checkpoint.py) | Authenticated opt-in normal-prior experiment checkpoints and frozen inputs. | `load_auxiliary`, `model_signature`, `load_components` |
| [normal_contact.py](../rl_sim/world_model/normal_contact.py) | Opt-in normal soft-prior correction; contact residual inputs remain39-D. | `normal_prior_increment`, `normal_contact_step`, `normal_contact_rollout`, `normal_contact_loss` |
| [normal_coupling.py](../rl_sim/world_model/normal_coupling.py) | Causal robot-conditioned normal prior, separate from contact39 features. | `acceleration_terms`, `own_robot_motion`, `normal_coupled_objective`, `make_predictor` |
| [normal_experiment.py](../rl_sim/world_model/normal_experiment.py) | Authorized normal-prior A/B: frozen base checks, then matched39-D refits. | `validate`, `fit`, `main` |
| [numerical_validation.py](../rl_sim/world_model/numerical_validation.py) | Task-scaled execution checks; solver working arrays remain diagnostics. | `assess_execution_difference` |
| [observation_engine.py](../rl_sim/world_model/observation_engine.py) | Known, causal camera module reused by learned and teacher worlds. | `CameraState`, `ObservationHistory`, `WorldObservationState`, `init_camera`, `advance_camera`, `advance_observation_history`, `init_world_observation`, `advance_world_observation`, `reset_world_observation`, `actor_observation`, `actor_observation_with_history` |
| [paired_attribution.py](../rl_sim/world_model/paired_attribution.py) | Frozen post-trial attribution helpers; never used by an optimizer or native step. | `return_components`, `orientation_margin_deg`, `flat_tree`, `gradient_statistics`, `target_from_checkpoint`, `ball_other_contacts`, `component_function` |
| [paired_history.py](../rl_sim/world_model/paired_history.py) | 640 ms of timestamp-paired ball/racket samples, with no dense actor history. | `FeedbackHistory`, `PairedHistory`, `init_feedback`, `init_history`, `_append`, `append_feedback`, `_aligned_pose`, `realign_pending`, `append_ball`, `encode_history`, `reset_selected` |
| [paired_precision.py](../rl_sim/world_model/paired_precision.py) | Private paired-training bindings; preserved production and pilot paths stay intact. | `random32_clone`, `learning_state`, `mixed_lambda_targets`, `compact_window`, `make_gradient`, `setup`, `signature`, `load_actor` |
| [persistent_pool.py](../rl_sim/world_model/persistent_pool.py) | Persistent versus short-branch state ownership; no implicit goal reset. | `PersistentPool`, `init_pool`, `add_count`, `count_as_int`, `prepare_window`, `commit_window`, `refresh_model` |
| [physical_events.py](../rl_sim/world_model/physical_events.py) | Contact and apex labels from actual 1 ms trajectories, independent of reward. | `PhysicalEventState`, `init_physical_events`, `physical_event_substep`, `scan_physical_events` |
| [physics_components.py](../rl_sim/world_model/physics_components.py) | Authenticate existing full-chain execution and39-D contact checkpoints. | `_authenticate_sources`, `validate_contact_conditioning`, `load_physics_components` |
| [physics_data.py](../rl_sim/world_model/physics_data.py) | Coherent command-conditioned physics windows; future states are labels only. | `read_physics_trial` |
| [physics_fitting.py](../rl_sim/world_model/physics_fitting.py) | Contact fitting with own-state full-chain robot predictions and fixed39 inputs. | `predicted_racket_motion`, `coupled_contact_objective` |
| [physics_metrics.py](../rl_sim/world_model/physics_metrics.py) | Same-clock coupled-physics errors; native labels select diagnostic events. | `ball_metrics`, `precontact_mask`, `robot_metrics` |
| [pilot_checkpoint.py](../rl_sim/world_model/pilot_checkpoint.py) | Pilot snapshots and explicit969-D frozen actor exports; no default promotion. | `pilot_signature`, `load_pilot_actor` |
| [pilot_contract.py](../rl_sim/world_model/pilot_contract.py) | Preregistered action diagnostics for the user-authorized WMJ-V1 pilot. | `counterfactual_actions`, `action_gate` |
| [pilot_learning.py](../rl_sim/world_model/pilot_learning.py) | Explicit bounded optimizer transactions; all regression rows are imagined. | `validate_learning_rows`, `optimize_pilot` |
| [pilot_runtime.py](../rl_sim/world_model/pilot_runtime.py) | Explicit candidate-only runtime; preserved native seeds create a fresh pool. | `validate_pilot`, `require_action_gate`, `setup_pilot`, `prepare_pilot_session` |
| [plan_training_seeds.py](../rl_sim/world_model/plan_training_seeds.py) | CPU-only offline admission and coverage report for existing training seeds. | `main` |
| [pose_margin.py](../rl_sim/world_model/pose_margin.py) | Default-off pose-margin reward candidate and frozen calibration mathematics. | `PoseMarginContract`, `roll_pitch_rad`, `pose_cost`, `penalty`, `augment_reward`, `coefficient_grid`, `select_weight`, `frozen_objectives` |
| [pose_trial.py](../rl_sim/world_model/pose_trial.py) | Isolated FP32 pose reward trial; historical runtimes stay unchanged. | `with_pose_reward`, `phase_plan`, `signature`, `verify_gate`, `load_actor` |
| [position_contact.py](../rl_sim/world_model/position_contact.py) | Opt-in 39-D contact: legacy19 plus q13 and delayed position-error7. | `PositionScale`, `position_features`, `position_scale_from_statistics`, `position_contact_step`, `position_contact_rollout`, `position_contact_loss` |
| [position_contact_data.py](../rl_sim/world_model/position_contact_data.py) | Position-only augmentation of coherent native contact data; no new replay. | `read_position_contact_trial`, `prepare_position_contact_batch`, `position_statistics_update`, `collect_position_statistics` |
| [preflight.py](../rl_sim/world_model/preflight.py) | Audit WMJ inputs and real-checkpoint migration without starting training. | `audit_actor`, `main` |
| [prepare_targeted_poses.py](../rl_sim/world_model/prepare_targeted_poses.py) | Offline independent flat-pose catalogue for one supplemental native batch. | `main` |
| [projection_consistency.py](../rl_sim/world_model/projection_consistency.py) | Private frozen projection probes and an opt-in quaternion FK candidate. | `capture_source`, `make_capture`, `make_quaternion_projector` |
| [projection_precision.py](../rl_sim/world_model/projection_precision.py) | Opt-in full-FP32 coordinate transforms for free-attitude projection only. | `_CoordinatePrecision`, `make_projector`, `native_projector`, `make_runtime` |
| [recover_reward_direction.py](../rl_sim/world_model/recover_reward_direction.py) | Complete only the authorized sampled probes with lossless bounded storage. | `write_gzip_json`, `sampled_collector`, `sampled_audit`, `gradient_recomputation_report`, `main` |
| [release_alignment.py](../rl_sim/world_model/release_alignment.py) | Ordinal release comparison with event-count changes and missing cases visible. | `compare_releases` |
| [render_collision_geometry.py](../rl_sim/world_model/render_collision_geometry.py) | Plot original mesh surfaces and the convex-hull cross-section of an alarm. | `_section`, `main` |
| [replay_overlay.py](../rl_sim/world_model/replay_overlay.py) | Lossless replay differences against immutable, authenticated teacher arrays. | `encode_overlay`, `decode_overlay` |
| [replay_support.py](../rl_sim/world_model/replay_support.py) | Augment immutable 128-control teacher trials with native support/FK labels. | `_restore`, `replay`, `main` |
| [resources.py](../rl_sim/world_model/resources.py) | Non-destructive storage and wall-clock limits shared by WMJ commands. | `StorageLimitError`, `StorageBudget`, `WallBudget` |
| [reward_direction.py](../rl_sim/world_model/reward_direction.py) | Private preparation, physical-event metrics and frozen reward comparisons. | `prepare_source`, `make_prepare`, `metric_task`, `observed_course`, `score_terms`, `summarize_stream`, `compare_returns` |
| [reward_direction_rollout.py](../rl_sim/world_model/reward_direction_rollout.py) | Same-state closed-loop native/model probes, with realized reward accounting. | `failure_code`, `make_native_step`, `make_world_step`, `collect_branches` |
| [robot_model.py](../rl_sim/world_model/robot_model.py) | 35->128->128->14 execution residual, conditioned on actual delayed commands. | `RobotScale`, `scale_from_training_statistics`, `robot_features`, `robot_residual_target`, `robot_step`, `robot_rollout` |
| [robot_support_model.py](../rl_sim/world_model/robot_support_model.py) | 47->128->128->26 robot execution model with measured support motion. | `expand_arm_parameters`, `support_features`, `support_residual_target`, `support_scale_from_statistics`, `support_robot_step`, `support_robot_rollout` |
| [run_expanded_auxiliary.py](../rl_sim/world_model/run_expanded_auxiliary.py) | Explicit bounded stages for the user-approved expanded auxiliary refresh. | `collect`, `audit_new`, `audit_diagnostic`, `main` |
| [runtime_guard.py](../rl_sim/world_model/runtime_guard.py) | Shared bounded single-GPU worker resources for WMJ fitting and evaluation. | `RunGuard` |
| [sampling.py](../rl_sim/world_model/sampling.py) | Four independent, reproducible CPU samplers with explicit draw probabilities. | `SamplingShortage`, `Draw`, `_counts`, `_shuffle`, `_balanced`, `sample_collection_regions`, `sample_model_windows`, `sample_branch_starts`, `sample_goals` |
| [search_frozen_initialization.py](../rl_sim/world_model/search_frozen_initialization.py) | Guarded offline search for flatter, natively warmed frozen-case seeds. | `angle_statistics`, `search`, `main` |
| [soft_contact.py](../rl_sim/world_model/soft_contact.py) | Finite 1 ms contact prior using native softness and a ball-side mass model. | `SoftContactContract`, `contract_from_collection`, `softness`, `SoftContactPrediction`, `soft_contact_increment` |
| [solver_auxiliary.py](../rl_sim/world_model/solver_auxiliary.py) | Independent 47-input diagnostic of native solver-stage racket J*qacc. | `execution_features`, `normal_projection_coefficients`, `scale_from_statistics`, `predict`, `supervised_loss`, `normal_velocity_error` |
| [state_adapter.py](../rl_sim/world_model/state_adapter.py) | Lossless numerical seeds bound to an authenticated static context. | `_leaf_signature`, `numeric_tree_digest`, `pack_numeric_state`, `restore_numeric_state`, `concatenate_phase_group` |
| [supplement_data.py](../rl_sim/world_model/supplement_data.py) | Read direct full-support supplement chunks without a duplicate replay overlay. | `diagnostic_batches`, `SupplementReader` |
| [support_data.py](../rl_sim/world_model/support_data.py) | Authenticated full-chain robot batches from coherent native replay overlays. | `SupportReplayReader`, `support_chunk`, `support_statistics_update`, `sample_support_windows` |
| [support_losses.py](../rl_sim/world_model/support_losses.py) | Thirteen-joint execution losses against native pose/twist supervision. | `racket_kinematics`, `support_execution_loss` |
| [tangent_checkpoint.py](../rl_sim/world_model/tangent_checkpoint.py) | Authenticated opt-in full-prior experiment checkpoints and frozen inputs. | `model_signature`, `load_components` |
| [tangent_contact.py](../rl_sim/world_model/tangent_contact.py) | Opt-in normal and tangential soft-prior correction; contact residual inputs remain39-D. | `tangent_prior_increment`, `tangent_contact_step`, `tangent_contact_rollout`, `tangent_contact_loss` |
| [tangent_coupling.py](../rl_sim/world_model/tangent_coupling.py) | Causal robot-conditioned normal and tangential prior, separate from contact39 features. | `tangent_coupled_objective`, `make_predictor` |
| [tangent_experiment.py](../rl_sim/world_model/tangent_experiment.py) | Authorized full-prior A/B: frozen base checks, then matched39-D refits. | `validate`, `fit`, `main` |
| [targeted_collection.py](../rl_sim/world_model/targeted_collection.py) | Independent spatial proposals for the bounded supplemental acquisition. | `missing_regions`, `acquisition_groups`, `independent_targets`, `targeted_draw`, `warm_fallback` |
| [targeted_recovery.py](../rl_sim/world_model/targeted_recovery.py) | Resume the same admitted acquisition; keep provenance and replay tolerances. | `compare_replay_arrays`, `restore_prepared` |
| [task.py](../rl_sim/world_model/task.py) | Batched goal accounting independent of robot, controller and camera state. | `TaskState`, `TaskEvent`, `init_task`, `record_release`, `record_apex`, `record_failure`, `switch_completed_goals` |
| [teacher_initialization.py](../rl_sim/world_model/teacher_initialization.py) | Explicit new-trial setup from an actually warmed V324 control chain. | `carry_warmed_controller`, `place_ball`, `prepare_warmup`, `camera_cache_fields`, `begin_ball_trial` |
| [teacher_trace.py](../rl_sim/world_model/teacher_trace.py) | Authenticated, private 1 kHz observation of the unchanged native physics loop. | `recorded_loop`, `contact_pair_masks`, `point_twist`, `_ObservePhysics`, `augment_source`, `_observe_native_loop`, `instrument_environment` |
| [train_critic_ema.py](../rl_sim/world_model/train_critic_ema.py) | Paired Critic EMA continuation with one identical frozen-world trajectory. | `failure_counts`, `shared_digest`, `check_plan`, `paired_result`, `run`, `main` |
| [train_critic_fit.py](../rl_sim/world_model/train_critic_fit.py) | Bounded existing-Critic fitting with a frozen Actor and structured world. | `take`, `storage_reservation`, `check_plan`, `make_forward`, `check_window`, `numerical_pair`, `restore_gate`, `evaluate`, `compare`, `run`, `main` |
| [train_fixed_critic.py](../rl_sim/world_model/train_fixed_critic.py) | Opt-in fixed-data value fitting; parent EMA labels never move. | `check_resource_continuation`, `check_plan`, `pair_rows`, `restore_gate`, `evaluate`, `compare`, `run`, `main` |
| [train_imagination.py](../rl_sim/world_model/train_imagination.py) | Bounded no-update validation of the pure imagined learner. | `validate_no_update_contract`, `distribution`, `tree_difference`, `gradient_summary`, `timed`, `validate`, `main` |
| [train_paired.py](../rl_sim/world_model/train_paired.py) | Opt-in, paired ten-update continuation with immutable no-update admission. | `phase_plan`, `phase_budget`, `assert_learning_fp32`, `restore_checks`, `validate`, `require_gates`, `train`, `main` |
| [train_pilot.py](../rl_sim/world_model/train_pilot.py) | At most20 S01 pure-imagination Actor updates, split by frozen native evaluations. | `validate_gradient_gate`, `gradient_window`, `train`, `main` |
| [train_pose_trial.py](../rl_sim/world_model/train_pose_trial.py) | Authorized FP32 pose-margin trial, at most ten Actor and forty Critic updates. | `phase_budget`, `assert_learning_fp32`, `restore_checks`, `validate`, `require_gates`, `train`, `main` |
| [training_seed_bank.py](../rl_sim/world_model/training_seed_bank.py) | Offline selection of independent, complete training-partition sources. | `initial_candidates`, `balanced_selection`, `incoming_admitted`, `read_initial_pose` |
| [trajectory_data.py](../rl_sim/world_model/trajectory_data.py) | CPU teacher chunks, stable group partitions and training-only statistics. | `_identifier`, `TrajectoryDataset`, `TrainNormalizer` |
| [velocity_reward.py](../rl_sim/world_model/velocity_reward.py) | Default-off inverse-flight outgoing-velocity reward and calibration helpers. | `VelocityRewardContract`, `velocity_costs`, `replacement_release_reward`, `clone_globals`, `candidate_course_step`, `diagnostic_course_step` |
| [world_control.py](../rl_sim/world_model/world_control.py) | Known WMJ control memory: clip, reference projection, native RMP and delays. | `ControlState`, `ControlContext`, `ControlContract`, `control_from_native`, `control_step` |
| [world_physics.py](../rl_sim/world_model/world_physics.py) | Own-state robot/FK/ball recursion using the approved47-D and39-D models. | `PhysicsState`, `_check_state`, `_physics_substep`, `physics_substep`, `physics_rollout`, `physics_control_step` |

## H. WMJ launcher索引

每个launcher均应先完整阅读。下列命令入口只是历史实现索引，**不是当前可直接运行列表**；已关闭plan会拒绝启动。涉及no-update/validate-only的标记不代表一定不用GPU或不写证据。

| Launcher（相对SIM） | 调用的WM入口 |
| --- | --- |
| [run_wmj_v1_action_effects.sh](../rl_sim/run_wmj_v1_action_effects.sh) | `world_model.audit_action_effects` |
| [run_wmj_v1_attribution_collision.sh](../rl_sim/run_wmj_v1_attribution_collision.sh) | `world_model.audit_attribution_collision` |
| [run_wmj_v1_attribution_modes.sh](../rl_sim/run_wmj_v1_attribution_modes.sh) | `world_model.audit_attribution_modes` |
| [run_wmj_v1_attribution_native.sh](../rl_sim/run_wmj_v1_attribution_native.sh) | `world_model.audit_attribution_native` |
| [run_wmj_v1_attribution_native_gradients.sh](../rl_sim/run_wmj_v1_attribution_native_gradients.sh) | `world_model.audit_attribution_native_gradients` |
| [run_wmj_v1_attribution_world.sh](../rl_sim/run_wmj_v1_attribution_world.sh) | `world_model.audit_attribution_world` |
| [run_wmj_v1_audit_contact_solver.sh](../rl_sim/run_wmj_v1_audit_contact_solver.sh) | `world_model.audit_contact_solver` |
| [run_wmj_v1_audit_error_attribution.sh](../rl_sim/run_wmj_v1_audit_error_attribution.sh) | `world_model.audit_error_attribution` |
| [run_wmj_v1_audit_free_imagination.sh](../rl_sim/run_wmj_v1_audit_free_imagination.sh) | `world_model.audit_free_imagination` |
| [run_wmj_v1_audit_orientation_initial.sh](../rl_sim/run_wmj_v1_audit_orientation_initial.sh) | `world_model.audit_orientation_initial` |
| [run_wmj_v1_audit_projection_seed.sh](../rl_sim/run_wmj_v1_audit_projection_seed.sh) | `world_model.audit_projection_seed` |
| [run_wmj_v1_audit_world_control.sh](../rl_sim/run_wmj_v1_audit_world_control.sh) | `world_model.audit_world_control_gpu` |
| [run_wmj_v1_auxiliary_refresh.sh](../rl_sim/run_wmj_v1_auxiliary_refresh.sh) |  |
| [run_wmj_v1_auxiliary_refresh_audit.sh](../rl_sim/run_wmj_v1_auxiliary_refresh_audit.sh) | `world_model.audit_auxiliary_refresh` |
| [run_wmj_v1_calibrate_pose_margin.sh](../rl_sim/run_wmj_v1_calibrate_pose_margin.sh) | `world_model.calibrate_pose_margin` |
| [run_wmj_v1_calibrate_velocity_reward.sh](../rl_sim/run_wmj_v1_calibrate_velocity_reward.sh) | `world_model.calibrate_velocity_reward` |
| [run_wmj_v1_collect.sh](../rl_sim/run_wmj_v1_collect.sh) | `world_model.collect_teacher` |
| [run_wmj_v1_corrected_candidate.sh](../rl_sim/run_wmj_v1_corrected_candidate.sh) | `world_model.audit_corrected_candidate` |
| [run_wmj_v1_corrected_reward.sh](../rl_sim/run_wmj_v1_corrected_reward.sh) | `world_model.audit_corrected_reward` |
| [run_wmj_v1_course_pool_no_update.sh](../rl_sim/run_wmj_v1_course_pool_no_update.sh) | `world_model.audit_course_pool` |
| [run_wmj_v1_critic_distribution.sh](../rl_sim/run_wmj_v1_critic_distribution.sh) | `world_model.audit_critic_distribution` |
| [run_wmj_v1_critic_ema.sh](../rl_sim/run_wmj_v1_critic_ema.sh) | `world_model.train_critic_ema` |
| [run_wmj_v1_critic_fit.sh](../rl_sim/run_wmj_v1_critic_fit.sh) | `world_model.train_critic_fit` |
| [run_wmj_v1_critic_inputs.sh](../rl_sim/run_wmj_v1_critic_inputs.sh) | `world_model.audit_critic_inputs` |
| [run_wmj_v1_critic_targets.sh](../rl_sim/run_wmj_v1_critic_targets.sh) | `world_model.audit_critic_targets` |
| [run_wmj_v1_evaluate_orientation_limit.sh](../rl_sim/run_wmj_v1_evaluate_orientation_limit.sh) | `world_model.evaluate_orientation_limit` |
| [run_wmj_v1_evaluate_paired.sh](../rl_sim/run_wmj_v1_evaluate_paired.sh) | `world_model.evaluate_paired` |
| [run_wmj_v1_evaluate_physics.sh](../rl_sim/run_wmj_v1_evaluate_physics.sh) | `world_model.evaluate_physics` |
| [run_wmj_v1_evaluate_pose_trial.sh](../rl_sim/run_wmj_v1_evaluate_pose_trial.sh) | `world_model.evaluate_pose_trial` |
| [run_wmj_v1_expanded_auxiliary.sh](../rl_sim/run_wmj_v1_expanded_auxiliary.sh) | `world_model.run_expanded_auxiliary` |
| [run_wmj_v1_fit_contact.sh](../rl_sim/run_wmj_v1_fit_contact.sh) | `world_model.fit_contact` |
| [run_wmj_v1_fit_physics_contact.sh](../rl_sim/run_wmj_v1_fit_physics_contact.sh) | `world_model.fit_physics_contact` |
| [run_wmj_v1_fit_robot.sh](../rl_sim/run_wmj_v1_fit_robot.sh) | `world_model.fit_robot` |
| [run_wmj_v1_fit_support.sh](../rl_sim/run_wmj_v1_fit_support.sh) | `world_model.fit_support` |
| [run_wmj_v1_fixed_critic.sh](../rl_sim/run_wmj_v1_fixed_critic.sh) | `world_model.train_fixed_critic` |
| [run_wmj_v1_fixed_critic_resume.sh](../rl_sim/run_wmj_v1_fixed_critic_resume.sh) | `world_model.train_fixed_critic` |
| [run_wmj_v1_frozen_evaluation.sh](../rl_sim/run_wmj_v1_frozen_evaluation.sh) | `world_model.evaluate_frozen` |
| [run_wmj_v1_full_prior_integration.sh](../rl_sim/run_wmj_v1_full_prior_integration.sh) | `world_model.audit_full_prior_world` |
| [run_wmj_v1_imagination_no_update.sh](../rl_sim/run_wmj_v1_imagination_no_update.sh) | `world_model.train_imagination` |
| [run_wmj_v1_initialization_search.sh](../rl_sim/run_wmj_v1_initialization_search.sh) | `world_model.search_frozen_initialization` |
| [run_wmj_v1_mixed_precision.sh](../rl_sim/run_wmj_v1_mixed_precision.sh) | `world_model.audit_mixed_precision` |
| [run_wmj_v1_module_gradients.sh](../rl_sim/run_wmj_v1_module_gradients.sh) | `world_model.audit_module_gradients` |
| [run_wmj_v1_new_pose_audit.sh](../rl_sim/run_wmj_v1_new_pose_audit.sh) | `world_model.audit_new_pose_world` |
| [run_wmj_v1_normal_components.sh](../rl_sim/run_wmj_v1_normal_components.sh) | `world_model.audit_normal_components` |
| [run_wmj_v1_normal_experiment.sh](../rl_sim/run_wmj_v1_normal_experiment.sh) | `world_model.normal_experiment` |
| [run_wmj_v1_paired.sh](../rl_sim/run_wmj_v1_paired.sh) | `world_model.train_paired` |
| [run_wmj_v1_paired_attribution.sh](../rl_sim/run_wmj_v1_paired_attribution.sh) | `world_model.audit_paired_attribution` |
| [run_wmj_v1_pilot_actions.sh](../rl_sim/run_wmj_v1_pilot_actions.sh) | `world_model.audit_pilot_actions` |
| [run_wmj_v1_pilot_evaluate.sh](../rl_sim/run_wmj_v1_pilot_evaluate.sh) | `world_model.evaluate_pilot` |
| [run_wmj_v1_pilot_gradients.sh](../rl_sim/run_wmj_v1_pilot_gradients.sh) | `world_model.audit_pilot_gradients` |
| [run_wmj_v1_pilot_train.sh](../rl_sim/run_wmj_v1_pilot_train.sh) | `world_model.train_pilot` |
| [run_wmj_v1_pose_trial.sh](../rl_sim/run_wmj_v1_pose_trial.sh) | `world_model.train_pose_trial` |
| [run_wmj_v1_projection_consistency.sh](../rl_sim/run_wmj_v1_projection_consistency.sh) | `world_model.audit_projection_consistency` |
| [run_wmj_v1_projection_precision.sh](../rl_sim/run_wmj_v1_projection_precision.sh) | `world_model.audit_projection_precision` |
| [run_wmj_v1_projection_precision_integrated.sh](../rl_sim/run_wmj_v1_projection_precision_integrated.sh) | `world_model.audit_projection_precision_integrated` |
| [run_wmj_v1_projection_variants.sh](../rl_sim/run_wmj_v1_projection_variants.sh) | `world_model.audit_projection_variants` |
| [run_wmj_v1_replay_support.sh](../rl_sim/run_wmj_v1_replay_support.sh) | `world_model.replay_support` |
| [run_wmj_v1_reward_direction.sh](../rl_sim/run_wmj_v1_reward_direction.sh) | `world_model.audit_reward_direction` |
| [run_wmj_v1_reward_direction_recover.sh](../rl_sim/run_wmj_v1_reward_direction_recover.sh) | `world_model.recover_reward_direction` |
| [run_wmj_v1_saved_pose_events.sh](../rl_sim/run_wmj_v1_saved_pose_events.sh) | `world_model.audit_saved_pose_events` |
| [run_wmj_v1_solver_auxiliary.sh](../rl_sim/run_wmj_v1_solver_auxiliary.sh) |  |
| [run_wmj_v1_tangent_components.sh](../rl_sim/run_wmj_v1_tangent_components.sh) | `world_model.audit_tangent_components` |
| [run_wmj_v1_tangent_diagnostic.sh](../rl_sim/run_wmj_v1_tangent_diagnostic.sh) | `world_model.audit_tangent_replay` |
| [run_wmj_v1_tangent_experiment.sh](../rl_sim/run_wmj_v1_tangent_experiment.sh) | `world_model.tangent_experiment` |
| [run_wmj_v1_targeted_collect.sh](../rl_sim/run_wmj_v1_targeted_collect.sh) | `world_model.collect_targeted` |
| [run_wmj_v1_targeted_pose_plan.sh](../rl_sim/run_wmj_v1_targeted_pose_plan.sh) | `world_model.prepare_targeted_poses` |
| [run_wmj_v1_targeted_world_audit.sh](../rl_sim/run_wmj_v1_targeted_world_audit.sh) | `world_model.audit_targeted_world` |
| [run_wmj_v1_training_seed_bank.sh](../rl_sim/run_wmj_v1_training_seed_bank.sh) | `world_model.build_training_seed_bank` |
| [run_wmj_v1_training_seed_plan.sh](../rl_sim/run_wmj_v1_training_seed_plan.sh) | `world_model.plan_training_seeds` |

## I. 测试导航

截至本次工作区有112个`test_wmj_v1_*.py`文件；这不是112项测试，也不表示本次全部执行。a52相关45项通过是2026-11-12历史证据。迁移后的实际执行结果重新登记。

| 方向 | 测试文件前缀（test目录） |
| --- | --- |
| 观测/状态 | test_wmj_v1_actor_migration, paired_history, observation, world_observation, state_adapter |
| 控制 | test_wmj_v1_world_control, control_projection, control_execution, rmp_lanes, projection_consistency |
| 几何/初始化 | test_wmj_v1_contact_surface, contact_kernel, mesh_geometry, initialization_geometry, frozen_initialization |
| 物理/模型 | test_wmj_v1_support_execution, solver_auxiliary, full_prior_world, ball_flight, physics_fitting |
| 课程/学习 | test_wmj_v1_course_env, course_rollout, course_schedule, imagined_learning, persistent_pool |
| 价值诊断 | test_wmj_v1_critic_features, critic_target_audit, critic_only_fit, critic_ema_pair, fixed_critic_fit, fixed_critic_resume, critic_input_audit |
| 资源与证据 | test_wmj_v1_memory_pressure, array_artifacts, validation_reservation, launcher_signals; test_training_memory_scope |

## J. 接手后禁止的捷径

- 不把a52报告完整保存等同于Critic或世界模型合格。
- 不用当前共享mjx_juggle_env.py替换被冻结V319/V324依赖路径。
- 不用配置中早期primary auxiliary u4096替代最近诊断u12288，或把u12288默认发布。
- 不把a51 checkpoint字段added512当绝对Critic步数512；正确端点为1656，Actor仍30。
- 不因CONTACT残差关闭删除其认证checkpoint；loader仍可能需要几何/归一化/来源合同。
- 不将短窗口、短分支命中率当16拍或连续4目标成功率。
- 不将终止bias下降当完整价值拟合通过；看RMSE、EV、来源尾部与删失。
- 不把分组近邻当完整Markov状态近邻，不把局部输入梯度当因果重要性。
- 不跳过cgroup、GPU归属或存储临时写入预留；不自动删除证据。
- 不批量改历史绝对路径或hash；跨根目录需显式映射及恢复验证。
- 不认为换机器、新分支或新会话继承了旧实验的剩余更新许可：a52结束，下一阶段QA未答。
