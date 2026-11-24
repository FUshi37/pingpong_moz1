"""Default-off calm velocity floor scored at every detached physical release."""

FIELDS=('reward/weak_calm_release_floor','release_floor/weak_event',
        'release_floor/minimum_vz_m_s','release_floor/predicted_apex_m')


def release_floor_terms(*, ball_z_m, ball_vz_m_s, gravity_m_s2, clean,
                        confirmed_hits, released, terminated, enabled=False,
                        weight=4., xp):
    """Return [N] diagnostics and bounded loss; no action or state correction.

    Calm legacy drag and native fluid are zero. Hence sqrt(2*g*(z_floor-z))
    is the exact vertical launch speed for the 1.405 m floor: the unchanged
    1.400 m confirmation lower edge plus a 5 mm training cushion. The native
    1.425 m precision target and every success threshold remain unchanged.
    """
    zero=xp.zeros_like(ball_z_m)
    if not enabled:
        return {name:zero for name in FIELDS}
    if weight not in (4.,12.):
        raise ValueError('Use a preregistered weak-release weight')
    valid=(xp.isfinite(ball_z_m)&xp.isfinite(ball_vz_m_s)&
           xp.isfinite(gravity_m_s2)&(gravity_m_s2>0.))
    active=(valid&clean&(confirmed_hits>=5)&released&~terminated&
            (ball_vz_m_s>0.))
    z=xp.where(active,ball_z_m,1.405)
    vz=xp.where(active,ball_vz_m_s,0.)
    gravity=xp.where(active,gravity_m_s2,9.81)
    minimum=xp.sqrt(2.*gravity*xp.maximum(1.405-z,0.))
    deficit=xp.maximum(minimum-vz,0.)/.030
    loss=xp.minimum(xp.sqrt(1.+deficit**2)-1.,10.)
    apex=z+vz**2/(2.*gravity)
    return dict(zip(FIELDS,(
        -weight*xp.where(active,loss,0.),
        (active&(deficit>0.)).astype(z.dtype),
        xp.where(active,minimum,0.),xp.where(active,apex,0.))))
