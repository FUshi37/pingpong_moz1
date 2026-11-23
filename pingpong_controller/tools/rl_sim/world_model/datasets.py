"""Stable group partitions and strict imagined-transition provenance."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import re


@dataclass
class SplitRegistry:
    """Append-only partition assignments across connected seed/session groups.

    A later edge between already differently assigned groups is rejected,
    rather than moving prior data or leaking an action fork into a holdout.
    Session identifiers must describe independent acquisition groups; a single
    giant session cannot supply independent training and holdout populations.
    """

    version: str
    assignments: dict[str, str] = field(default_factory=dict)

    def assign(self, *, root_seed_family: str, episode: str, session: str) -> str:
        if not all(isinstance(v, str) and v for v in (root_seed_family, episode, session)):
            raise ValueError('All grouping identifiers must be nonempty strings')
        # Length prefixes avoid ambiguous concatenation of user identifiers.
        keys = [f'root:{root_seed_family}', f'session:{session}',
                f'episode:{len(session)}:{session}:{episode}']
        known = {self.assignments[k] for k in keys if k in self.assignments}
        if len(known) > 1:
            raise ValueError('Cross-split group connection would cause data leakage')
        if known:
            split = known.pop()
        else:
            raw = f'{self.version}\0{root_seed_family}'.encode('utf-8')
            value = int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') / 2**64
            split = 'train' if value < .8 else 'validation' if value < .9 else 'test'
        self.assignments.update({k: split for k in keys})
        return split

    def to_dict(self) -> dict:
        return dict(version=self.version, assignments=copy.deepcopy(self.assignments))

    @classmethod
    def from_dict(cls, data: dict) -> SplitRegistry:
        assignments = data['assignments']
        if not all(v in ('train', 'validation', 'test') for v in assignments.values()):
            raise ValueError('Unknown data partition')
        return cls(data['version'], dict(assignments))


def require_imagination(provenance: dict) -> None:
    """Reject teacher transitions or real returns at actor/critic entrypoints."""
    if provenance.get('origin') != 'imagination':
        raise ValueError('Actor/critic updates require imagination transitions exclusively')
    revision = provenance.get('model_revision')
    if (not isinstance(revision, int) or isinstance(revision, bool) or revision < 1
            or not re.fullmatch('[0-9a-f]{64}', provenance.get('model_sha256', ''))):
        raise ValueError('Imagination must identify a frozen dynamics checkpoint')
    if provenance.get('return_origin', 'imagination') != 'imagination':
        raise ValueError('Critic returns must come exclusively from imagination')
