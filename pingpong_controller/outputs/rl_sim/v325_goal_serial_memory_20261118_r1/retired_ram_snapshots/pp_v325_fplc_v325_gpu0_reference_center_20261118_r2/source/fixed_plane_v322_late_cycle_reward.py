"""Opt-in bounded contact-centering and apex-contraction event costs."""

LATE_CYCLE_SETTINGS = dict(center_free_m=.0045, center_scale_m=.0065,
    center_max_cost=4., contraction_ratio=.8, apex_free_m=.005,
    apex_scale_m=.035, apex_max_cost=12.)
LATE_CENTER_PRECISION_SETTINGS = {**LATE_CYCLE_SETTINGS,
    'center_max_cost': 32., 'apex_max_cost': 0.}


def late_cycle_reward_terms(metrics: dict, previous_hits, *, xp, settings=None) -> dict:
    """Price native confirmed contacts after the first three, arrays (N,).

    Previous/current signed errors come from native physical apex history.
    Invalid history is masked before arithmetic; no native input is mutated.
    """
    cfg = LATE_CYCLE_SETTINGS if settings is None else settings
    event = (metrics['hit_event_count'] > 0.) & (previous_hits >= 3)
    distance = metrics['hit_contact_center_dist_sum'] / xp.maximum(
        metrics['hit_event_count'], 1.)
    center = xp.clip((distance-cfg['center_free_m'])/cfg['center_scale_m'], 0., 1.)
    valid = metrics['v322_previous_apex_valid'] > 0.
    previous = xp.where(valid, metrics['v322_previous_apex_error_m'], 0.)
    current = xp.where(event & valid, metrics['v322_current_apex_error_m'], 0.)
    expansion = xp.maximum(xp.abs(current)-cfg['contraction_ratio']
        * xp.abs(previous)-cfg['apex_free_m'], 0.)
    energy = xp.clip(expansion/cfg['apex_scale_m'], 0., 1.)
    return {
        'reward/v322_late_center_penalty': xp.where(
            event, -cfg['center_max_cost']*center**2, 0.),
        'reward/v322_late_energy_contraction_penalty': xp.where(
            event & valid, -cfg['apex_max_cost']*energy**2, 0.),
    }
