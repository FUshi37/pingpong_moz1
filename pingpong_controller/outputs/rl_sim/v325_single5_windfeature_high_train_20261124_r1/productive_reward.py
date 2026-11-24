"""Bounded wind objective with real-flight timing and unchanged physical gates."""


def safe_return_site_z(mode, background_m_s, anchor_z_m, gain, bounds_m, *, xp):
    """Use the nearest safe anchor, rather than a sudden midpoint-height target."""
    speed=xp.sqrt(xp.sum(background_m_s[..., :2]**2,axis=-1))
    margin=.010+.015*xp.clip(speed/1.6,0.,1.)**2
    safe=xp.clip(anchor_z_m,bounds_m[0]+margin,bounds_m[1]-margin)
    target=anchor_z_m+gain*(safe-anchor_z_m)
    return xp.where(mode>0,target,.5*(bounds_m[0]+bounds_m[1]))


def outgoing_quality_cost(error_m_s, *, xp):
    """Bound below4 with nonconstant tails; vertical precision remains primary."""
    x=xp.abs(error_m_s/xp.asarray([.035,.035,.020]))
    huber=.5*xp.minimum(x,1.)**2+xp.maximum(x-1.,0.)
    loss=huber[...,0]+huber[...,1]+2.*huber[...,2]
    return 4.*loss/(1.+loss)


def productive_reward(native, m, gain, *, dt_s, xp):
    """Preserve calm exactly; charge release errors without positive release credit."""
    recent=m['dynamics/recent']>.5
    failed=m['terminated']>.5
    hit=m['hit_event_count']
    interval=xp.clip(m['dynamics/hit_interval_s'],0.,1.2)
    center=m['hit_contact_center_dist_sum']/xp.maximum(hit,1.)
    vxy=m['reference_motion/recurrent_excess_vxy_sq_sum']/xp.maximum(m['recurrent_hit_event_count'],1.)
    omega=m['reference_motion/steady_excess_omega_sq_sum']/xp.maximum(m['rapid_steady_steady_events'],1.)
    quality=1./(1.+(center/.006)**2+vxy/.055**2+omega/.28**2)
    error=xp.stack([m[f'dynamics/outgoing_error_v{axis}_m_s'] for axis in 'xyz'],axis=-1)
    cost=outgoing_quality_cost(error,xp=xp)
    cost=xp.where(m['dynamics/reference_valid']>.5,cost,4.)
    apex=m['wind_height/apex_events']
    apex_cost=m['reference_quality/apex_error_sq_sum_m2']/xp.maximum(apex,1.)/.008**2
    pieces={
        'alive':20.*dt_s*recent,
        'hit':8.*interval*hit*quality*recent,
        'outgoing':-interval*m['dynamics/release_event']*cost,
        'normal':2.*dt_s*m['dynamics/precontact']/(1.+(m['dynamics/normal_error_deg']/1.5)**2),
        'height_margin':5.*dt_s*xp.clip(m['dynamics/z_margin_m']/.025,0.,1.),
        'braking':-4.*dt_s*xp.clip(m['dynamics/braking_cost'],0.,4.),
        'apex':2.*xp.clip(m['dynamics/period_s'],.2,1.2)*apex/(1.+apex_cost),
    }
    target=sum(pieces.values())
    target=xp.where(failed,xp.minimum(native,-500.),target)
    result=xp.where(gain>0.,native+gain*(target-native),native)
    pieces={name:gain*xp.where(failed,0.,value) for name,value in pieces.items()}
    pieces['failure']=gain*xp.where(failed,xp.minimum(native,-500.),0.)
    return result,pieces
