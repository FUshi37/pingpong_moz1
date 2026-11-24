"""Small additive onset/precision terms; keep the better native S15 objective."""
FIELDS=tuple('reward/targeted_'+name for name in ('precontact','brake','normal','vertical','apex','termination_reserve','outgoing_xy','outgoing_z'))


def targeted_terms(m,wind_gain,clean_gain,post_onset_hits,*,enabled=False,variant='brake_first_precision',failure_reserve=0.,xp):
    if not enabled:return {}
    onset=1.+2.*xp.exp(-xp.minimum(post_onset_hits,50.)/4.)
    gain=wind_gain+clean_gain
    live=m['terminated']<.5
    def huber(x):return xp.sqrt(1.+x*x)-1.
    if variant in ('kinetic_brake','calm_precision'):
        vz=m['diagnostic/racket_vz_m_s'];speed=xp.abs(vz)
        z=m['diagnostic/racket_z_preview_m']-.08*vz
        clearance=xp.where(vz<0.,z-1.,1.15-z)
        required=speed**2/(2.*xp.maximum(clearance-.005,.002))
        braking=xp.maximum(-xp.sign(vz)*m['diagnostic/racket_az_m_s2'],0.)
        gate=(speed>.05)&(clearance<.04)&(clearance/xp.maximum(speed,.001)<.2)
        deficit=xp.maximum(required-braking,0.)/2.
        cost=-20.*.005*gain*onset*gate*xp.minimum(huber(deficit),6.)
        out={name:xp.zeros_like(m['terminated']) for name in FIELDS}
        out['reward/targeted_brake']=xp.where(live,cost,0.)
        out['reward/targeted_termination_reserve']=-failure_reserve*m['terminated']
        if variant=='calm_precision':
            release=live*(m['dynamics/release_event']>.5)*(m['dynamics/reference_valid']>.5)*clean_gain
            error=lambda axis:xp.where(release>0.,m['dynamics/outgoing_error_v'+axis+'_m_s'],0.)
            xy=huber(error('x')/.035)+huber(error('y')/.035)
            vertical=huber(error('z')/.030)
            out['reward/targeted_outgoing_xy']=-2.*release*xp.minimum(xy,10.)
            out['reward/targeted_outgoing_z']=-2.*release*xp.minimum(vertical,10.)
        return out
    preview_error=xp.maximum(m['diagnostic/predicted_center_m']-.004,0.)/.010
    lower,upper=1.015,1.135
    vz=m['diagnostic/racket_vz_m_s']
    # Extend the recorded80ms preview to120ms for early outward braking.
    z=m['diagnostic/racket_z_preview_m']+.04*vz
    brake=xp.maximum(lower-z,0.)*(vz<0.)+xp.maximum(z-upper,0.)*(vz>0.)
    period=xp.clip(m['dynamics/period_s'],.4,.6)
    vertical=xp.minimum((m['dynamics/outgoing_error_vz_m_s']/.020)**2,9.)
    apex=m['reference_quality/apex_error_sq_sum_m2']/xp.maximum(m['wind_height/apex_events'],1.)/.010**2
    values=(
        -4.*.005*gain*onset*m['diagnostic/precontact']*xp.minimum(huber(preview_error),6.),
        -80.*.005*gain*onset*xp.minimum((brake/.010)**2,6.),
        2.*.005*wind_gain*m['dynamics/precontact']*xp.exp(-.5*(m['dynamics/normal_error_deg']/2.)**2),
        -.5*wind_gain*m['dynamics/release_event']*m['dynamics/reference_valid']*vertical,
        2.*period*wind_gain*m['wind_height/apex_events']/(1.+apex))
    if variant not in ('brake_first_precision','brake_only'):raise ValueError(variant)
    out={name:xp.zeros_like(m['terminated']) for name in FIELDS}
    out.update({name:xp.where(live,value if variant!='brake_only' or name.endswith('_brake') else xp.zeros_like(value),0.) for name,value in zip(FIELDS[:5],values)})
    out['reward/targeted_termination_reserve']=-failure_reserve*m['terminated']
    return out
