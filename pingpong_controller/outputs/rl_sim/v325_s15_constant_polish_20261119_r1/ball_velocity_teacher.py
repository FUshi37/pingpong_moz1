"""Default-off simulation-only instantaneous ball-velocity teacher suffix."""
import jax.numpy as jnp
from frozen_calm_wind_teacher_retry1 import TeacherObservationEnv


class VelocityTeacherObservationEnv(TeacherObservationEnv):
    """Actor: original114 + wind4 + true world ball velocity3/3m/s =121."""
    def __init__(self,inner,enabled=False):
        super().__init__(inner,enabled=enabled)
        self.obs_dim=121 if enabled else inner.obs_dim

    def observe(self,state,obs):
        result=super().observe(state,obs)
        if not self.enabled:
            return result
        velocity=state.data.qvel[:,self.ball_vadr:self.ball_vadr+3]/3.
        return jnp.concatenate([result,velocity],axis=-1)
