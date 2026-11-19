"""V325/V326 tail rewards aligned with center, band and survival gates."""
from dataclasses import field, make_dataclass
from functools import lru_cache
import math

REVISION = 's8_reward_bridges_20261113'


def repair_contract() -> dict:
    """Describe additive costs; physical events and all target gates stay native."""
    return dict(revision=REVISION, extra_center_multiplier=1.,
                extra_survival_multiplier=.5, out_of_band_cost=12.,
                physical_failure_cost=20., first_seven_lessons_unchanged=True)


def repair_scale(lesson, version: int) -> float:
    if version not in (325, 326):
        return 0.
    return float(hasattr(lesson,'pose_target_deg') or any(
        token in lesson.name for token in ('_bridge', '_s08_', '_s09_', '_pose')))


@lru_cache(maxsize=4)
def _config_type(parent_type: type) -> type:
    return make_dataclass('WindRewardRepairConfig',
        [('wind_reward_scale', float, field(default=1.))], bases=(parent_type,), frozen=True)


def with_reward_repair(cfg, scale: float):
    if not math.isfinite(scale) or not 0. <= scale <= 1.:
        raise ValueError('Wind reward scale must lie in [0,1]')
    if scale == 0.:
        return cfg
    return _config_type(type(cfg))(**vars(cfg), wind_reward_scale=scale)


def reward_terms(metrics: dict, *, scale: float, xp) -> dict:
    """Per-lane arrays: pay only recent real juggling; charge physical failure.

    Native band flags count every eligible physical outcome, including weak
    contacts. Natural time-limit truncation receives no failure cost. Center
    retains its native linear slope; no extra positive per-hit credit is added.
    """
    return {
        'reward/wind_repair_center': scale * metrics[
            'reward/fixed_plane_v159_mean_center_penalty'],
        'reward/wind_repair_survival': scale * .5 * metrics[
            'reward/fixed_contact_survival_reward'],
        'reward/wind_repair_band': -scale * 12. * xp.maximum(
            metrics['v92_late_contact_event'] - metrics['v92_in_1p4_1p45_event'], 0.),
        'reward/wind_repair_failure': -scale * 20. * metrics['terminated'],
    }


def reward_environment(base: type) -> type:
    """Apply after the native long-episode survival adapter, with exact legacy off."""
    class RewardEnv(base):
        def step(self, state, action):
            after, obs, reward, done, metrics = super().step(state, action)
            scale = getattr(self.cfg, 'wind_reward_scale', 0.)
            if scale == 0.:
                return after, obs, reward, done, metrics
            import jax.numpy as jnp
            terms = reward_terms(metrics, scale=scale, xp=jnp)
            if hasattr(self.cfg,'wind_pose'):
                from fixed_plane_wind_pose import scheduled_quality_scale,shelter_gain,pose_clock_steps
                from fixed_plane_wind_objectives import objective_terms
                cfg=self.cfg.wind_pose
                strength=scheduled_quality_scale(pose_clock_steps(state,cfg),cfg,xp=jnp)*shelter_gain(state,self.dt,xp=jnp)
                allowance=0.
                if cfg.strong_wind_reward:
                    from fixed_plane_wind_strong import apex_target_reward
                    from fixed_plane_wind_precision import precision_terms
                    allowance=metrics['wind/required_outgoing_vxy_m_s']
                    terms['reward/wind_apex_target']=strength*apex_target_reward(metrics,xp=jnp)
                    # Native late-impact eligibility starts at hit4. Wind's
                    # five-hit shelter must not suppress these outcome losses.
                    terms.update(precision_terms(metrics,gain=jnp.ones_like(state.step_count,dtype=jnp.float32),
                        projection_gain=(state.hit_count>0).astype(jnp.float32),xp=jnp))
                terms.update(objective_terms(metrics,version=cfg.objective_version,scale=strength,xp=jnp,
                                             ball_allowance=allowance))
                metrics['pose/objective_scale']=strength
            reward = reward + sum(terms.values())
            metrics.update(terms)
            if getattr(self.cfg, 'wind_gate_precision_positive', False):
                from fixed_plane_wind_gate_precision import normalized_clean_reward
                from fixed_plane_wind_pose import shelter_gain
                clean_gain = shelter_gain(state,self.dt,xp=jnp)*(state.wind_mode == 0)
                normalized = normalized_clean_reward(reward,metrics['terminated'],
                    clean_gain=clean_gain,xp=jnp)
                metrics['reward/gate_clean_rescale'] = normalized-reward
                reward = normalized
            if getattr(self.cfg, 'wind_bounded_recovery_reward', False):
                from fixed_plane_wind_bounded_recovery_reward import (
                    bounded_reward, racket_z_margin_score, Z_MARGIN_SCORE_CONTRACT,
                    MOTION_PRIORITY_SCORE_CONTRACT, Z_ANTICIPATION_SCORE_CONTRACT,
                    APEX_PRECISION_SCORE_CONTRACT, NATIVE_Z_FAILURE_SCORE_CONTRACT,
                    AUXILIARY_APEX_SCORE_CONTRACT,
                    anticipatory_racket_z_margin_score)
                from fixed_plane_wind_pose import shelter_gain
                gain = shelter_gain(state, self.dt, xp=jnp)
                from fixed_plane_wind_gate_precision import recovery_gain
                gain = recovery_gain(gain, state.wind_mode,
                    enabled=getattr(self.cfg, 'wind_gate_precision', False))
                interval = (state.step_count+1)*self.dt-state.last_counted_hit_time
                score_mode = getattr(self.cfg,'wind_recovery_quality_score_mode','gaussian')
                z_score = None
                if score_mode in (Z_MARGIN_SCORE_CONTRACT['identity'],
                                  MOTION_PRIORITY_SCORE_CONTRACT['identity']):
                    z_score = racket_z_margin_score(metrics['racket_z'],
                        self.cfg.fixed_contact_racket_z_bounds_m,
                        Z_MARGIN_SCORE_CONTRACT['racket_z_margin_m'], xp=jnp)
                elif score_mode in (Z_ANTICIPATION_SCORE_CONTRACT['identity'],
                                    APEX_PRECISION_SCORE_CONTRACT['identity'],
                                    NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                                    AUXILIARY_APEX_SCORE_CONTRACT['identity']):
                    bounds = self.cfg.fixed_contact_racket_z_bounds_m
                    margin = Z_ANTICIPATION_SCORE_CONTRACT['racket_z_margin_m']
                    vz_m_s = after.prev_racket_vel[:, 2]
                    z_score = anticipatory_racket_z_margin_score(metrics['racket_z'],
                        vz_m_s, bounds, margin,
                        Z_ANTICIPATION_SCORE_CONTRACT['racket_z_prediction_horizon_s'], xp=jnp)
                    metrics['recovery/z_position_margin_score'] = racket_z_margin_score(
                        metrics['racket_z'], bounds, margin, xp=jnp)
                    metrics['recovery/racket_vz_m_s'] = vz_m_s
                reward, shaped_terms = bounded_reward(reward, metrics, gain=gain,
                    interval_s=interval, dt_s=self.dt, xp=jnp,
                    aggregation=getattr(self.cfg,'wind_recovery_quality_aggregation','arithmetic'),
                    score_mode=score_mode, z_margin_score=z_score)
                metrics.update(shaped_terms)
            if getattr(self.cfg, 'wind_dynamics_guidance', False):
                from fixed_plane_wind_dynamics_guidance import apply_guidance
                reward, metrics = apply_guidance(self, state, after, reward, metrics)
            metrics['reward/total'] = reward
            return after, obs, reward, done, metrics
    return RewardEnv
