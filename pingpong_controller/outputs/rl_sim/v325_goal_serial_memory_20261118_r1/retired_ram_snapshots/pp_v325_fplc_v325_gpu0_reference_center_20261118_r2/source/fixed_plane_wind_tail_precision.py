"""Explicit wind-only tail precision; historical reward modes remain unchanged."""
from __future__ import annotations

from typing import Any


def precision_contract() -> dict:
    return dict(revision='independent_vertical_tail_precision_v1',
        outgoing_sigma_m_s=[.035, .035, .030],
        horizontal='sum of Huber X/Y losses; independent cap4',
        vertical='half squared normalized Z error; independent cap8',
        outgoing_weight=4., failure_floor=-500.,
        height_recovery='continuous racket-Z midpoint cost;2/s times squared error/25mm,cap9',
        failure='retain any larger native penalty; every real termination retained',
        discount_utility='at dt=.005 and gamma=.9995, -20/s continuation is -200, above -500 failure',
        scope='wind after existing five-hit shelter and1s ramp only',
        measured_height='same observed-apex band, weak-flight accounting and numerical gates',
        unchanged='normal reference,3D shooting,return point,braking,clean reward,physics,observations,PPO,gates')


def outgoing_precision_loss(error_m_s: Any, *, xp: Any) -> dict:
    """Compute (...,3) velocity loss without XY errors masking Z precision."""
    error = error_m_s / xp.asarray([.035, .035, .030])
    absolute = xp.abs(error[..., :2])
    huber = .5 * xp.minimum(absolute, 1.) ** 2 + xp.maximum(absolute - 1., 0.)
    horizontal = xp.minimum(xp.sum(huber, axis=-1), 4.)
    vertical = .5 * xp.minimum(xp.square(error[..., 2]), 16.)
    return dict(total=horizontal + vertical, horizontal=horizontal, vertical=vertical)


def failure_cost(terminated: Any, native_racket_penalty: Any, *, xp: Any) -> Any:
    """Charge actual failures without making a physical termination cheaper."""
    return xp.where(terminated, xp.minimum(-500., native_racket_penalty), 0.)


def height_recovery_cost(z_m: Any, target_z_m: Any, *, dt_s: float, xp: Any) -> Any:
    """Anchor the fixed contact-height cycle, including arbitrarily slow drift."""
    return 2. * dt_s * xp.minimum(xp.square((z_m - target_z_m) / .025), 9.)
