"""Per-impact apex precision for the explicit full20-degree wind tail."""

REVISION = 'full20_eligible4_feasible_proposals_20261114'


def precision_contract() -> dict:
    return dict(revision=REVISION,target_apex_m=1.425,
        quadratic_cost_per_m2=40000.,quadratic_cost_cap=24.,extra_out_of_band_cost=48.,
        timing='native eligible fourth-plus impacts; independent of wind shelter',
        projection_extra_multiplier=9.,projection_timing='after first confirmed hit; native correction cost only',
        purpose='price apex dispersion even when its population mean is correct',
        archived_prefix_unchanged=True)


def precision_terms(metrics, *, gain, xp, projection_gain=None) -> dict:
    event=metrics['rapid_steady_steady_events']
    apex=metrics['rapid_steady_steady_apex_sum_z_m']/xp.maximum(event,1.)
    result = dict(
        **{'reward/wind_apex_precision':-gain*event*xp.minimum(40000.*(apex-1.425)**2,24.)},
        **{'reward/wind_band_commit':-gain*48.*xp.maximum(
            metrics['v92_late_contact_event']-metrics['v92_in_1p4_1p45_event'],0.)})
    if projection_gain is not None:
        result['reward/wind_feasible_command'] = projection_gain*9.*xp.minimum(
            metrics['reward/cartesian_pose_projection_penalty'],0.)
    return result
