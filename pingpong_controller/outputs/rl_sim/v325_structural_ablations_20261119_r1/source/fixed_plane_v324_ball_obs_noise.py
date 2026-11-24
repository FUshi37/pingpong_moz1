"""V324 registration: preserve M13 and add only M10's camera-position noise."""
from fixed_plane_v323_ball_obs_noise import spec_for

SPEC = spec_for(324)
PROFILE = SPEC.profile
PARENT_SHA256 = SPEC.parent_sha256
COURSE = SPEC.course
