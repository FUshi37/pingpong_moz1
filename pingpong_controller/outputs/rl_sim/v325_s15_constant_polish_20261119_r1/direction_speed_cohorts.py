"""Opt-in constant-wind cohorts; native observations and physics stay unchanged."""
import jax
import jax.numpy as jnp
import numpy as np

from fixed_plane_v325_wind import _draw

SECTORS = 8


def wind_sector(background):
    angle = jnp.arctan2(-jnp.abs(background[..., 1]), background[..., 0])
    return jnp.clip(jnp.floor((angle + jnp.pi) / jnp.pi * SECTORS), 0, SECTORS-1).astype(jnp.int32)


def cohort_labels(n_envs):
    lane = jnp.arange(n_envs)
    clean = lane < round(.1*n_envs)
    return clean, jnp.mod(lane-round(.1*n_envs), SECTORS)


def peak_cohort_mask(n_envs):
    clean, sectors = cohort_labels(n_envs)
    wind_slot = jnp.maximum(jnp.arange(n_envs)-round(.1*n_envs), 0)
    return (~clean) & ((wind_slot//SECTORS+sectors)%4 == 0)


def keys_for_cohorts(keys, cfg):
    clean, sectors = cohort_labels(keys.shape[0])
    peak_slots = peak_cohort_mask(keys.shape[0])

    def matches(candidate):
        u = _draw(candidate, 32501, 8)
        angle = 2.*jnp.pi*u[:, 1]
        direction = jnp.stack((jnp.cos(angle), -jnp.abs(jnp.sin(angle)), jnp.zeros_like(angle)), axis=-1)
        return jnp.where(clean, u[:, 0] < cfg.mode_weights[0],
                         (u[:, 0] >= cfg.mode_weights[0]) & (wind_sector(direction) == sectors) &
                         ((_draw(candidate, 32505, 1)[:, 0] < cfg.peak_wind_fraction) == peak_slots))

    def draw(carry):
        count, accepted, chosen = carry
        proposed = jax.vmap(jax.random.fold_in)(keys, jnp.full(keys.shape[0], count))
        good = matches(proposed)
        return count+1, accepted | good, jnp.where((~accepted & good)[:, None], proposed, chosen)

    return jax.lax.while_loop(lambda x: ~jnp.all(x[1]), draw,
        (jnp.asarray(1, jnp.uint32), matches(keys), keys))[2]


class DirectionEnv:
    """Sampler and extra read-only diagnostics; no actor or control changes."""
    def __init__(self, inner, balanced=False):
        self.inner, self.balanced = inner, balanced

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def keys_for(self, keys):
        return keys_for_cohorts(keys, self.cfg.wind_config) if self.balanced else keys

    def reset(self, keys):
        return self.inner.reset(self.keys_for(keys))

    def reset_done(self, state, obs, done, keys):
        return self.inner.reset_done(state, obs, done, self.keys_for(keys))

    def get_critic_obs(self, state, obs):
        return self.inner.get_critic_obs(state, obs)

    def step(self, state, action):
        after, obs, reward, done, metrics = self.inner.step(state, action)
        sector = wind_sector(state.wind_background_m_s)
        metrics['polish/peak_assigned'] = ((state.wind_mode > 0) & (jnp.linalg.norm(state.wind_background_m_s,axis=-1) >= self.cfg.wind_config.max_speed_m_s-1.e-5)).astype(jnp.float32)
        metrics['polish/wind_sector'] = jnp.where(state.wind_mode == 0, -1, sector).astype(jnp.float32)
        # This measures actual native collision normals, unlike the old face metric.
        contact = after.data._impl.contact
        pair = (((contact.geom[..., 0] == self.ball_geom_id) & (contact.geom[..., 1] == self.racket_geom_id)) |
                ((contact.geom[..., 1] == self.ball_geom_id) & (contact.geom[..., 0] == self.racket_geom_id)))
        force = jnp.zeros_like(contact.dist)
        for index, (address, dim) in enumerate(zip(np.asarray(contact.efc_address), np.asarray(contact.dim))):
            if address >= 0:
                count = 1 if dim == 1 else 2*(dim-1)
                force = force.at[:, index].set(jnp.sum(after.data._impl.efc_force[:, address:address+count], axis=-1))
        valid = pair & (contact.dist < 0.) & (force > 1.e-6)
        normal = contact.frame[..., 0, :]
        normal = jnp.where(normal[..., 2:3] < 0., -normal, normal)
        face = after.data.site_xmat[:, self.racket_site_id].reshape((-1, 3, 3))[:, :, 2]
        face = jnp.where(face[:, 2:3] < 0., -face, face)
        deviation = jnp.rad2deg(jnp.arccos(jnp.clip(jnp.sum(normal*face[:, None, :], axis=-1), -1., 1.)))
        metrics['polish/physical_contact_frames'] = jnp.any(valid, axis=-1).astype(jnp.float32)
        metrics['polish/edge_contact_frames'] = jnp.any(valid & (deviation > 10.), axis=-1).astype(jnp.float32)
        metrics['polish/contact_face_deviation_max_deg'] = jnp.max(jnp.where(valid, deviation, 0.), axis=-1)
        return after, obs, reward, done, metrics
