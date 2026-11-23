"""Recorded-reference diagnostics separating projection from RMP arithmetic.

Teacher references are diagnostic inputs only. None of these paths implements
or qualifies a corrected imagined rollout.
"""

from contextlib import nullcontext

import jax
import jax.numpy as jnp

from recovered_rmp_jax import joint_rmp_step
from .control_projection import project_reference
from .control_rmp import apply_output_delay, rmp_step_lanes


def make_component_replays(chain, contract):
    c = contract.projection

    @jax.jit
    def project(context, actions, measured, references):
        def tick(unused, sample):
            action, q, reference = sample
            denominator = c.velocity_limit[None]*contract.action_velocity_scale*context.action_scale_mult[:, None]
            velocity = jnp.clip(jnp.clip(action,-1.,1.)*denominator,-c.velocity_limit[None],c.velocity_limit[None])
            result = project_reference(chain,c,q,reference,velocity)
            return unused, dict(reference=result.reference_q, fallback=result.fallback_to_hold,
                accepted=jnp.clip(result.accepted_qvel/jnp.maximum(denominator,1e-6),-1.,1.))
        return jax.lax.scan(tick, None, (actions, measured, references))[1]

    def make_rmp(*, batched=False, highest=False):
        @jax.jit
        def replay(initial, context, references):
            if batched:
                initial = initial._replace(rmp=initial.rmp._replace(
                    target_buffer_index=initial.rmp.target_buffer_index[0],
                    output_buffer_index=initial.rmp.output_buffer_index[0]))

            def tick(state, reference):
                precision = jax.default_matmul_precision('highest') if highest else nullcontext()
                with precision:
                    if batched:
                        rmp,q,dq,ddq = joint_rmp_step(state.rmp,reference,c.q_lower,c.q_upper,c.velocity_limit,
                            contract.acceleration_limit_rad_s2,contract.jerk_limit_rad_s3,contract.rmp_params,.001,
                            **context.rmp,servo_steps=5,target_update_delay_steps=3,use_safety_bounds=True,
                            use_jerk_limit=False,return_history=True)
                        commands = dict(q=q,dq=dq,ddq=ddq)
                    else:
                        rmp,commands = rmp_step_lanes(state.rmp,reference,context.rmp,q_min=c.q_lower,q_max=c.q_upper,
                            velocity_limit=c.velocity_limit,acceleration_limit=contract.acceleration_limit_rad_s2,
                            jerk_limit=contract.jerk_limit_rad_s3,params=contract.rmp_params)
                delayed,buffers = {},{}
                for key in ('q','dq','ddq'):
                    delayed[key],buffers[key] = apply_output_delay(state.output_delay[key],commands[key],context.output_delay_steps)
                return state._replace(reference_q_rad=reference,rmp=rmp,output_delay=buffers),delayed
            return jax.lax.scan(tick,initial,references)[1]
        return replay
    return dict(projection=project, rmp_lanes=make_rmp(), rmp_lanes_highest=make_rmp(highest=True),
                native_batched_rmp=make_rmp(batched=True))
