"""Interpret native constraint forces with the model's actual friction cone."""


def contact_normal_forces(force_n, addresses, dimensions, cone: str, *, xp):
    """Map [...,constraint] forces to [...,contact] normal forces in newtons."""
    if cone not in ('elliptic', 'pyramidal'):
        raise ValueError('Unknown friction cone')
    values = []
    for address, dimension in zip(addresses, dimensions):
        address, dimension = int(address), int(dimension)
        if address < 0:
            values.append(xp.zeros_like(force_n[..., 0]))
            continue
        count = 1 if cone == 'elliptic' or dimension == 1 else 2*(dimension-1)
        if address+count > force_n.shape[-1]:
            raise ValueError('Contact force window outside native constraint array')
        values.append(xp.sum(force_n[..., address:address+count], axis=-1))
    return xp.stack(values, axis=-1)
