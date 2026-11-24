"""User-selected physical wind domain; unrestricted historical paths stay exact."""
from dataclasses import field,make_dataclass
from functools import lru_cache
from typing import Any


def direction_domain_contract() -> dict:
    return dict(revision='horizontal_negative_y_v1',allowed='world horizontal Y<=0',
        task_boundary='user-reported real-robot joint-limit feasibility',
        sampling='reflect positive Y to negative Y; uniform lower-half-plane angles, unchanged speed',
        realization='reflect the final OU/gust/background sum before applying the startup gain',
        references='same projection in theoretical outgoing-velocity wind forecasts',
        increments='OU and gust vectors are latent increments, not standalone physical air velocities',
        unchanged='wind speed range,episode duration,physics force law,control,reward,observations,optimizer and gates')


@lru_cache(maxsize=8)
def _config_type(parent: type) -> type:
    return make_dataclass('NegativeYWindConfig',
        [('wind_y_nonpositive',bool,field(default=True))],bases=(parent,),frozen=True)


def with_negative_y_config(cfg: Any):
    if getattr(cfg,'wind_y_nonpositive',False):
        return cfg
    return _config_type(type(cfg))(**vars(cfg),wind_y_nonpositive=True)


def constrain_wind(vector: Any,cfg: Any,*,xp: Any):
    """Reflect [...,3] physical wind without changing its speed or allowed vectors."""
    if not getattr(cfg,'wind_y_nonpositive',False):
        return vector
    y=xp.where(vector[...,1]>0.,-vector[...,1],vector[...,1])
    return xp.stack((vector[...,0],y,vector[...,2]),axis=-1)
