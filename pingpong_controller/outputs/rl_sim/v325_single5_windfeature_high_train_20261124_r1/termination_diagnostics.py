"""All native termination terms for the authenticated fixed-contact profile."""

NATIVE_DONE_FIELDS = tuple('done/' + name for name in (
    'ball_too_low', 'ball_too_high', 'ball_x_out_of_bounds', 'ball_y_out_of_bounds',
    'ball_view_x_too_low', 'ball_view_x_too_high',
    'ball_view_y_too_low', 'ball_view_y_too_high',
    'ball_view_z_too_low', 'ball_view_z_too_high',
    'base_x_out_of_bounds', 'base_y_out_of_bounds', 'base_z_out_of_bounds',
    'base_roll_out_of_bounds', 'base_pitch_out_of_bounds',
    'racket_too_far_from_anchor', 'racket_too_high', 'racket_too_low',
    'racket_x_out_of_bounds', 'racket_y_out_of_bounds', 'racket_z_out_of_bounds',
    'late_weak_streak', 'pending_hit_recontact', 'failed_hit_confirmation',
    'hit_racket_vxy_exceeded', 'hit_vxy_exceeded', 'hit_racket_up_cos_exceeded',
    'hit_curriculum_complete', 'fixed_contact_first_hit_timeout',
    'fixed_contact_recurrent_hit_timeout', 'fixed_contact_arrival_complete',
    'fixed_contact_arrived_hit_complete', 'fixed_contact_post_hit_recovery_complete',
    'fixed_contact_return_basin_complete',
))
TERMINATION_FIELDS = NATIVE_DONE_FIELDS + ('fixed_contact_predictor_fail_safe',)
