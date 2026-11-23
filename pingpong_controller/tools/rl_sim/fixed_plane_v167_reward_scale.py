"""Uniform reward scale with unchanged physical metrics and component ratios."""
import math


def scale_reward_terms(reward, terms, *, multiplier: float):
    """Scale final reward and every reward/ leaf, preserving all other metrics."""
    factor=float(multiplier)
    if not math.isfinite(factor) or factor<=0.:
        raise ValueError('Reward multiplier must be finite and positive')
    return reward*factor,{key:(value*factor if key.startswith('reward/') else value)
                          for key,value in terms.items()}
