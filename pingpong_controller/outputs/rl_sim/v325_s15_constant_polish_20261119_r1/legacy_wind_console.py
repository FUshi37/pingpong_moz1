"""Original V325 terminal format, using the original read-only statistics."""
from fixed_plane_wind_training_gates import TrainingSegments
from fixed_plane_v325_wind import MODE_NAMES
from fixed_plane_wind_reference_constraints import reference_quality_summary


class LegacyWindConsole:
    """Keep console statistics separate from learning and progression gates."""

    def __init__(self, n_envs: int, quality_fields: tuple[str, ...], seconds: float = 180.):
        self.fields = tuple(quality_fields)
        self.tracker = TrainingSegments(seconds, .005, n_envs,
                                        quality_fields=self.fields)

    def add(self, metrics: dict, done: object) -> None:
        self.tracker.add(hits=metrics['new_hit'], done=done,
            terminated=metrics['terminated'], recent=metrics['v321/recent_hit'],
            modes=metrics['wind/mode'], quality_metrics={
                name: metrics[name] for name in self.fields})

    def line(self, update: int, hold: int, gate: dict) -> str:
        totals = {field: sum(row[mode][field]
            for row in self.tracker.qualities for mode in MODE_NAMES)
            for field in self.fields}
        quality = reference_quality_summary(totals)
        blockers = (' blockers='+','.join(gate['failures'])
                    if not gate['passed'] else '')
        return (f'[V325] S15/32 u={update} hold={hold}/16 '
                f'wind={self.tracker.wind_summary()} quality={quality} Adam+32 '
                f'gates_passed={int(gate["passed"])}{blockers}')
