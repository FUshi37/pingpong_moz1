"""Read-only cone-aware replacement; do not reinterpret older force logs."""
import jax.numpy as jnp
import mujoco
import numpy as np

from direction_speed_cohorts import DirectionEnv, wind_sector
from native_contact_force import contact_normal_forces


class CorrectContactEnv(DirectionEnv):
    """Keep native reset/state/obs/reward/done, correcting only extra logs."""

    def __init__(self, inner, balanced=False):
        super().__init__(inner, balanced)
        self.force_cone = ('elliptic' if int(self.mj_model.opt.cone) ==
                           int(mujoco.mjtCone.mjCONE_ELLIPTIC) else 'pyramidal')

    def step(self, state, action):
        after, obs, reward, done, metrics = self.inner.step(state, action)
        sector = wind_sector(state.wind_background_m_s)
        metrics['polish/peak_assigned'] = ((state.wind_mode > 0) & (
            jnp.linalg.norm(state.wind_background_m_s, axis=-1) >=
            self.cfg.wind_config.max_speed_m_s-1.e-5)).astype(jnp.float32)
        metrics['polish/wind_sector'] = jnp.where(state.wind_mode == 0, -1, sector).astype(jnp.float32)
        contact = after.data._impl.contact
        force = contact_normal_forces(after.data._impl.efc_force,
            np.asarray(contact.efc_address), np.asarray(contact.dim), self.force_cone, xp=jnp)
        ball_first = ((contact.geom[..., 0] == self.ball_geom_id) &
                      (contact.geom[..., 1] == self.racket_geom_id))
        ball_second = ((contact.geom[..., 1] == self.ball_geom_id) &
                       (contact.geom[..., 0] == self.racket_geom_id))
        valid = (ball_first | ball_second) & (contact.dist < 0.) & (force > 1.e-6)
        normal = jnp.where(ball_first[..., None], -contact.frame[..., 0, :], contact.frame[..., 0, :])
        face = after.data.site_xmat[:, self.racket_site_id].reshape((-1, 3, 3))[:, :, 2]
        deviation = jnp.rad2deg(jnp.arccos(jnp.clip(
            jnp.sum(normal*face[:, None, :], axis=-1), -1., 1.)))
        metrics['polish/physical_contact_frames'] = jnp.any(valid, axis=-1).astype(jnp.float32)
        metrics['polish/edge_contact_frames'] = jnp.any(valid & (deviation > 10.), axis=-1).astype(jnp.float32)
        metrics['polish/contact_face_deviation_max_deg'] = jnp.max(jnp.where(valid, deviation, 0.), axis=-1)
        metrics['polish/cone_aware_logger'] = jnp.ones_like(reward)
        return after, obs, reward, done, metrics
