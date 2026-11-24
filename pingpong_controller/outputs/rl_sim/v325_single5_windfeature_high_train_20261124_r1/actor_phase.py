"""Use native on-policy group weights without changing policy inputs or returns."""


def phase_update(native, *, minimum_mass=0.):
    if not minimum_mass:
        return native
    if not 0. < minimum_mass < 1.:
        raise ValueError('Invalid startup actor mass')
    def update(learner, runner, transitions):
        startup = transitions.obs[..., 112] < .999
        grouped = transitions._replace(metrics={**transitions.metrics,
            'reset_falling_contact_active': startup.astype('float32')})
        candidate, metrics = native(learner, runner, grouped)
        return candidate, {**metrics,
            'actor_phase/startup_fraction': startup.mean(),
            'actor_phase/minimum_mass': minimum_mass}
    return update
