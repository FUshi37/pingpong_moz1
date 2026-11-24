"""Explicit recovery-only online recipe; old V325 checkpoints keep their course."""
from dataclasses import asdict, replace
from fixed_plane_wind_bounded_recovery_reward import (
    GATE_MARGIN_SCORE_CONTRACT, GATE_MARGIN_GAUSSIAN_APEX_SCORE_CONTRACT,
    HEIGHT_PRIORITY_SCORE_CONTRACT, Z_MARGIN_SCORE_CONTRACT,
    MOTION_PRIORITY_SCORE_CONTRACT, Z_ANTICIPATION_SCORE_CONTRACT,
    APEX_PRECISION_SCORE_CONTRACT, NATIVE_Z_FAILURE_SCORE_CONTRACT,
    AUXILIARY_APEX_SCORE_CONTRACT)
from fixed_plane_wind_history_optimizer import HISTORY_XY_OPTIMIZER_CONTRACT


RECIPE_ID = 'recovery_full20_stage_anchor_002_v1'
RECIPE = dict(identity=RECIPE_ID, pose_start_deg=20., pose_ramp_updates=0,
              actor_anchor_kl_coef=.02, anchor='current stage entry policy')
BOUNDED_RECIPE_ID = 'recovery_full20_bounded_quality_anchor002_v1'
BOUNDED_RECIPE = {**RECIPE, 'identity': BOUNDED_RECIPE_ID,
                  'reward': 'bounded_quality_real_survival_v1'}
RECIPES = {RECIPE_ID: RECIPE, BOUNDED_RECIPE_ID: BOUNDED_RECIPE}
JOINT_RECIPE_ID = 'recovery_full20_joint_quality_anchor002_v1'
JOINT_RECIPE = {**BOUNDED_RECIPE, 'identity': JOINT_RECIPE_ID,
                'quality_aggregation': 'geometric'}
RECIPES[JOINT_RECIPE_ID] = JOINT_RECIPE
JOINT_SOURCE_SHA256 = '288e00cab64dbff4c47327b668e8f0a645b2065279f1a23764cfee2acd8e911b'
PRECISION_RECIPE_ID = 'recovery_full20_joint_quality_sigma01_anchor005_v1'
PRECISION_RECIPE = {**JOINT_RECIPE, 'identity': PRECISION_RECIPE_ID,
                    'actor_anchor_kl_coef': .005, 'exploration_max_std': .01}
RECIPES[PRECISION_RECIPE_ID] = PRECISION_RECIPE
PRECISION_SOURCE_SHA256 = 'ad139462afac9fa9054f7e69d84e8909c5bde083b0d30ed23701d26e3a7763ba'
LOW_LR_RECIPE_ID = 'recovery_full20_joint_precision_low_lr_v1'
LOW_LR_RECIPE = {**PRECISION_RECIPE, 'identity': LOW_LR_RECIPE_ID,
                 'learning_rate': 7.5e-6}
RECIPES[LOW_LR_RECIPE_ID] = LOW_LR_RECIPE
LOW_LR_SOURCE_SHA256 = 'eba6a6c3a090539f0c643fbf1170d376a6d80e0c12cd25c483b8bd26165693e0'
GATE_MARGIN_RECIPE_ID = 'recovery_full20_gate_margin_quality_v1'
GATE_MARGIN_RECIPE = {**LOW_LR_RECIPE, 'identity': GATE_MARGIN_RECIPE_ID,
    'quality_score_mode': GATE_MARGIN_SCORE_CONTRACT['identity'],
    'quality_score_contract': dict(GATE_MARGIN_SCORE_CONTRACT)}
RECIPES[GATE_MARGIN_RECIPE_ID] = GATE_MARGIN_RECIPE
GATE_MARGIN_SOURCE_SHA256 = '0f305d90b82dffdadb03d31bd539e4ca2117c8030a368f8cdf8c621397563c82'
GAUSSIAN_APEX_RECIPE_ID = 'recovery_full20_gate_margin_gaussian_apex_v1'
GAUSSIAN_APEX_RECIPE = {**GATE_MARGIN_RECIPE, 'identity': GAUSSIAN_APEX_RECIPE_ID,
    'quality_score_mode': GATE_MARGIN_GAUSSIAN_APEX_SCORE_CONTRACT['identity'],
    'quality_score_contract': dict(GATE_MARGIN_GAUSSIAN_APEX_SCORE_CONTRACT)}
RECIPES[GAUSSIAN_APEX_RECIPE_ID] = GAUSSIAN_APEX_RECIPE
GAUSSIAN_APEX_SOURCE_SHA256 = 'bbe90e8a3a0c3a4e8020e2be82f8dddda6770c45ab8330c1e624cd76860ddfdd'
HEIGHT_PRIORITY_RECIPE_ID = 'recovery_full20_height_priority_quality_v1'
HEIGHT_PRIORITY_RECIPE = {**GAUSSIAN_APEX_RECIPE, 'identity': HEIGHT_PRIORITY_RECIPE_ID,
    'quality_score_mode': HEIGHT_PRIORITY_SCORE_CONTRACT['identity'],
    'quality_score_contract': dict(HEIGHT_PRIORITY_SCORE_CONTRACT)}
RECIPES[HEIGHT_PRIORITY_RECIPE_ID] = HEIGHT_PRIORITY_RECIPE
HEIGHT_PRIORITY_SOURCE_SHA256 = '9e48a0b0bbed884cfbb0f3d624780b7639dc1d91a425cedcc4ee2b1a66bc7a45'
FINE_PRECISION_RECIPE_ID = 'recovery_full20_height_sigma005_anchor00125_v1'
FINE_PRECISION_RECIPE = {**HEIGHT_PRIORITY_RECIPE, 'identity': FINE_PRECISION_RECIPE_ID,
    'exploration_max_std': .005, 'learning_rate': 3.75e-6,
    'actor_anchor_kl_coef': .00125,
    'precision_coupling': 'LR/sigma and anchor/(2*sigma^2) unchanged from R14'}
RECIPES[FINE_PRECISION_RECIPE_ID] = FINE_PRECISION_RECIPE
FINE_PRECISION_SOURCE_SHA256 = '93ab99e5be5fb905a50fda0a18942af439bb3be6ab21932117fdb521994886e9'
Z_MARGIN_RECIPE_ID = 'recovery_full20_height_z_margin_v1'
Z_MARGIN_RECIPE = {**FINE_PRECISION_RECIPE, 'identity': Z_MARGIN_RECIPE_ID,
    'quality_score_mode': Z_MARGIN_SCORE_CONTRACT['identity'],
    'quality_score_contract': dict(Z_MARGIN_SCORE_CONTRACT)}
RECIPES[Z_MARGIN_RECIPE_ID] = Z_MARGIN_RECIPE
Z_MARGIN_SOURCE_SHA256 = '11a922e69e89ee982f40650114334f1e55ec5f2265298142dc53dc987fcd401f'
MOTION_PRIORITY_RECIPE_ID = 'recovery_full20_height_motion_z_margin_v1'
MOTION_PRIORITY_RECIPE = {**Z_MARGIN_RECIPE, 'identity': MOTION_PRIORITY_RECIPE_ID,
    'quality_score_mode': MOTION_PRIORITY_SCORE_CONTRACT['identity'],
    'quality_score_contract': dict(MOTION_PRIORITY_SCORE_CONTRACT)}
RECIPES[MOTION_PRIORITY_RECIPE_ID] = MOTION_PRIORITY_RECIPE
MOTION_PRIORITY_SOURCE_SHA256 = '8d375402f96cbf7a809bb5b752f2931990205a843c22aeeaf08f58b2fc7e5f40'
Z_ANTICIPATION_RECIPE_ID = 'recovery_full20_height_motion_z_anticipation_v1'
Z_ANTICIPATION_RECIPE = {**MOTION_PRIORITY_RECIPE, 'identity': Z_ANTICIPATION_RECIPE_ID,
    'quality_score_mode': Z_ANTICIPATION_SCORE_CONTRACT['identity'],
    'quality_score_contract': dict(Z_ANTICIPATION_SCORE_CONTRACT)}
RECIPES[Z_ANTICIPATION_RECIPE_ID] = Z_ANTICIPATION_RECIPE
Z_ANTICIPATION_SOURCE_SHA256 = '8077b6be377c660be870fe36d4b483d4aec9b3fd0a9fa360f7541aa9c3558a5a'
APEX_PRECISION_RECIPE_ID = 'recovery_full20_height_motion_z_anticipation_apex008_v1'
APEX_PRECISION_RECIPE = {**Z_ANTICIPATION_RECIPE, 'identity': APEX_PRECISION_RECIPE_ID,
    'quality_score_mode': APEX_PRECISION_SCORE_CONTRACT['identity'],
    'quality_score_contract': dict(APEX_PRECISION_SCORE_CONTRACT)}
RECIPES[APEX_PRECISION_RECIPE_ID] = APEX_PRECISION_RECIPE
APEX_PRECISION_SOURCE_SHA256 = '35eb9135e652be9df94a6e1b3582153c7d5dfe2767e8a4093f322e5a7c941d5b'
NATIVE_Z_FAILURE_RECIPE_ID = 'recovery_full20_apex008_native_z_failure_floor_v1'
NATIVE_Z_FAILURE_RECIPE = {**APEX_PRECISION_RECIPE, 'identity': NATIVE_Z_FAILURE_RECIPE_ID,
    'quality_score_mode': NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
    'quality_score_contract': dict(NATIVE_Z_FAILURE_SCORE_CONTRACT)}
RECIPES[NATIVE_Z_FAILURE_RECIPE_ID] = NATIVE_Z_FAILURE_RECIPE
NATIVE_Z_FAILURE_SOURCE_SHA256 = '8ae81d926d0d1b9e34e8a22531b2985a15f1fd21ee795622a4979b422f9df0a2'
AUXILIARY_APEX_RECIPE_ID = 'recovery_full20_native_z_floor_apex_support_v1'
AUXILIARY_APEX_RECIPE = {**NATIVE_Z_FAILURE_RECIPE, 'identity': AUXILIARY_APEX_RECIPE_ID,
    'quality_score_mode': AUXILIARY_APEX_SCORE_CONTRACT['identity'],
    'quality_score_contract': dict(AUXILIARY_APEX_SCORE_CONTRACT)}
RECIPES[AUXILIARY_APEX_RECIPE_ID] = AUXILIARY_APEX_RECIPE
AUXILIARY_APEX_SOURCE_SHA256 = '01ba3e0111c097d78b0bcf17ea8ed92a64d54f1777c9313d316af2027b31bc29'
HISTORY_XY_RECIPE_ID = 'recovery_full20_apex_support_history_xy_eps000002_v1'
HISTORY_XY_RECIPE = {**AUXILIARY_APEX_RECIPE, 'identity': HISTORY_XY_RECIPE_ID,
    'history_xy_optimizer': dict(HISTORY_XY_OPTIMIZER_CONTRACT)}
RECIPES[HISTORY_XY_RECIPE_ID] = HISTORY_XY_RECIPE
HISTORY_XY_SOURCE_SHA256 = '3a8767c53fcf7b048d9dbca1002f9396321fda1070339074824aa74d7bd0e72d'


def prepare_recipe_migration(stack, payload, resume):
    """Allow each changed recipe only from its reviewed complete learner."""
    import math
    from fixed_plane_v323_ball_obs_noise import digest_file
    contract = payload.get('wind_contract', {})
    stack.online_recipe_migration = contract.get('online_recipe_migration')
    requested = getattr(stack, 'requested_online_recipe', None)
    if requested == HISTORY_XY_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == HISTORY_XY_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != AUXILIARY_APEX_RECIPE
                or resume is None or digest_file(resume) != HISTORY_XY_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (14, 9100, 2802450432, 793775)
                or payload.get('stage_converged', False)):
            raise ValueError('History XY optimizer requires the pinned full R22 S15 learner')
        migration = dict(source_sha256=HISTORY_XY_SOURCE_SHA256, source_update=9100,
            source_true_step=2802450432, source_adam_t=793775,
            previous_recipe=AUXILIARY_APEX_RECIPE_ID, new_recipe=HISTORY_XY_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='Adam epsilon 2e-6 only for14 nonzero-lag residual-history XY rows; initial full learner/reward/observation/gates exact')
        stack.online_recipe_migration = migration
        return {**payload, 'wind_contract': {**contract,
            'bootstrap_online_recipe': HISTORY_XY_RECIPE, 'online_recipe_migration': migration}}
    if requested == AUXILIARY_APEX_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == AUXILIARY_APEX_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != NATIVE_Z_FAILURE_RECIPE
                or resume is None or digest_file(resume) != AUXILIARY_APEX_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (14, 9000, 2789343232, 790575)
                or payload.get('stage_converged', False)):
            raise ValueError('Apex support requires the pinned full R21 S15 learner')
        migration = dict(source_sha256=AUXILIARY_APEX_SOURCE_SHA256, source_update=9000,
            source_true_step=2789343232, source_adam_t=790575,
            previous_recipe=NATIVE_Z_FAILURE_RECIPE_ID, new_recipe=AUXILIARY_APEX_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='independent bounded 4-per-second steady apex event channel only; learner/gates exact')
        stack.online_recipe_migration = migration
        return {**payload, 'env_cfg': {**payload['env_cfg'],
                'wind_recovery_quality_score_mode': AUXILIARY_APEX_SCORE_CONTRACT['identity']},
                'wind_contract': {**contract, 'bootstrap_online_recipe': AUXILIARY_APEX_RECIPE,
                                  'online_recipe_migration': migration}}
    if requested == NATIVE_Z_FAILURE_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == NATIVE_Z_FAILURE_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != APEX_PRECISION_RECIPE
                or resume is None or digest_file(resume) != NATIVE_Z_FAILURE_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (14, 8700, 2750021632, 780975)
                or payload.get('stage_converged', False)):
            raise ValueError('Native Z failure floor requires the pinned full R20 S15 learner')
        migration = dict(source_sha256=NATIVE_Z_FAILURE_SOURCE_SHA256, source_update=8700,
            source_true_step=2750021632, source_adam_t=780975,
            previous_recipe=APEX_PRECISION_RECIPE_ID, new_recipe=NATIVE_Z_FAILURE_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='retain existing native Z failure cost below -100 only; positive rewards/learner/gates exact')
        stack.online_recipe_migration = migration
        return {**payload, 'env_cfg': {**payload['env_cfg'],
                'wind_recovery_quality_score_mode': NATIVE_Z_FAILURE_SCORE_CONTRACT['identity']},
                'wind_contract': {**contract, 'bootstrap_online_recipe': NATIVE_Z_FAILURE_RECIPE,
                                  'online_recipe_migration': migration}}
    if requested == APEX_PRECISION_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == APEX_PRECISION_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != Z_ANTICIPATION_RECIPE
                or resume is None or digest_file(resume) != APEX_PRECISION_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (14, 8600, 2736914432, 777775)
                or payload.get('stage_converged', False)):
            raise ValueError('Apex precision requires the pinned full R19 S15 learner')
        migration = dict(source_sha256=APEX_PRECISION_SOURCE_SHA256, source_update=8600,
            source_true_step=2736914432, source_adam_t=777775,
            previous_recipe=Z_ANTICIPATION_RECIPE_ID, new_recipe=APEX_PRECISION_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='apex Gaussian sigma 0.0125m to 0.008m only; full learner and gates exact')
        stack.online_recipe_migration = migration
        return {**payload, 'env_cfg': {**payload['env_cfg'],
                'wind_recovery_quality_score_mode': APEX_PRECISION_SCORE_CONTRACT['identity']},
                'wind_contract': {**contract, 'bootstrap_online_recipe': APEX_PRECISION_RECIPE,
                                  'online_recipe_migration': migration}}
    if requested == Z_ANTICIPATION_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == Z_ANTICIPATION_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != MOTION_PRIORITY_RECIPE
                or resume is None or digest_file(resume) != Z_ANTICIPATION_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (13, 7700, 2618949632, 748975)
                or payload.get('stage_converged', False)):
            raise ValueError('Z-anticipation reward requires the pinned full R18 S14 learner')
        migration = dict(source_sha256=Z_ANTICIPATION_SOURCE_SHA256, source_update=7700,
            source_true_step=2618949632, source_adam_t=748975,
            previous_recipe=MOTION_PRIORITY_RECIPE_ID, new_recipe=Z_ANTICIPATION_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='20ms outward native-Z momentum reward anticipation only; full learner and gates exact')
        stack.online_recipe_migration = migration
        return {**payload, 'env_cfg': {**payload['env_cfg'],
                'wind_recovery_quality_score_mode': Z_ANTICIPATION_SCORE_CONTRACT['identity']},
                'wind_contract': {**contract, 'bootstrap_online_recipe': Z_ANTICIPATION_RECIPE,
                                  'online_recipe_migration': migration}}
    if requested == MOTION_PRIORITY_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == MOTION_PRIORITY_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != Z_MARGIN_RECIPE
                or resume is None or digest_file(resume) != MOTION_PRIORITY_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (13, 7272, 2562850816, 735279)
                or payload.get('stage_converged', False)):
            raise ValueError('Motion-priority reward requires the pinned full R17 S14 learner')
        migration = dict(source_sha256=MOTION_PRIORITY_SOURCE_SHA256, source_update=7272,
            source_true_step=2562850816, source_adam_t=735279,
            previous_recipe=Z_MARGIN_RECIPE_ID, new_recipe=MOTION_PRIORITY_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='fourth-root racket XY/recurrent mean focus on hit credit only; full learner and gates exact')
        stack.online_recipe_migration = migration
        return {**payload, 'env_cfg': {**payload['env_cfg'],
                'wind_recovery_quality_score_mode': MOTION_PRIORITY_SCORE_CONTRACT['identity']},
                'wind_contract': {**contract, 'bootstrap_online_recipe': MOTION_PRIORITY_RECIPE,
                                  'online_recipe_migration': migration}}
    if requested == Z_MARGIN_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == Z_MARGIN_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != FINE_PRECISION_RECIPE
                or resume is None or digest_file(resume) != Z_MARGIN_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (13, 6600, 2474770432, 713775)
                or payload.get('stage_converged', False)):
            raise ValueError('Z-margin reward requires the pinned full R15 S14 learner')
        migration = dict(source_sha256=Z_MARGIN_SOURCE_SHA256, source_update=6600,
            source_true_step=2474770432, source_adam_t=713775,
            previous_recipe=FINE_PRECISION_RECIPE_ID, new_recipe=Z_MARGIN_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='15mm native racket-Z positive-credit margin only; full learner, anchor and gates exact')
        stack.online_recipe_migration = migration
        return {**payload, 'env_cfg': {**payload['env_cfg'],
                'wind_recovery_quality_score_mode': Z_MARGIN_SCORE_CONTRACT['identity']},
                'wind_contract': {**contract, 'bootstrap_online_recipe': Z_MARGIN_RECIPE,
                                  'online_recipe_migration': migration}}
    if requested == FINE_PRECISION_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == FINE_PRECISION_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != HEIGHT_PRIORITY_RECIPE
                or resume is None or digest_file(resume) != FINE_PRECISION_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (13, 6000, 2396127232, 694575)
                or payload.get('stage_converged', False)):
            raise ValueError('Fine precision requires the pinned full R14 S14 learner')
        migration = dict(source_sha256=FINE_PRECISION_SOURCE_SHA256, source_update=6000,
            source_true_step=2396127232, source_adam_t=694575,
            previous_recipe=HEIGHT_PRIORITY_RECIPE_ID, new_recipe=FINE_PRECISION_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='measured sigma cap .005; LR and anchor scaled to preserve Gaussian step/pull; full learner exact')
        stack.online_recipe_migration = migration
        return {**payload, 'args': {**payload['args'], 'max_log_std': math.log(.005),
                'learning_rate': 3.75e-6, 'actor_anchor_kl_coef': .00125},
                'wind_contract': {**contract, 'bootstrap_online_recipe': FINE_PRECISION_RECIPE,
                                  'online_recipe_migration': migration}}
    if requested == HEIGHT_PRIORITY_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == HEIGHT_PRIORITY_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != GAUSSIAN_APEX_RECIPE
                or resume is None or digest_file(resume) != HEIGHT_PRIORITY_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (12, 5500, 2330591232, 678575)
                or payload.get('stage_converged', False)):
            raise ValueError('Height-priority reward requires the pinned full R13 learner')
        migration = dict(source_sha256=HEIGHT_PRIORITY_SOURCE_SHA256, source_update=5500,
            source_true_step=2330591232, source_adam_t=678575,
            previous_recipe=GAUSSIAN_APEX_RECIPE_ID, new_recipe=HEIGHT_PRIORITY_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='bounded height-focus factor only; full learner, stage anchor and gates exact')
        stack.online_recipe_migration = migration
        return {**payload, 'env_cfg': {**payload['env_cfg'],
                'wind_recovery_quality_score_mode': HEIGHT_PRIORITY_SCORE_CONTRACT['identity']},
                'wind_contract': {**contract, 'bootstrap_online_recipe': HEIGHT_PRIORITY_RECIPE,
                                  'online_recipe_migration': migration}}
    if requested == GAUSSIAN_APEX_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == GAUSSIAN_APEX_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != GATE_MARGIN_RECIPE
                or resume is None or digest_file(resume) != GAUSSIAN_APEX_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (12, 5000, 2265055232, 662575)
                or payload.get('stage_converged', False)):
            raise ValueError('Gaussian apex repair requires the pinned full R12 learner')
        migration = dict(source_sha256=GAUSSIAN_APEX_SOURCE_SHA256, source_update=5000,
            source_true_step=2265055232, source_adam_t=662575,
            previous_recipe=GATE_MARGIN_RECIPE_ID, new_recipe=GAUSSIAN_APEX_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='original Gaussian apex score only; other margin scores, learner and anchor exact')
        stack.online_recipe_migration = migration
        return {**payload, 'env_cfg': {**payload['env_cfg'],
                'wind_recovery_quality_score_mode': GATE_MARGIN_GAUSSIAN_APEX_SCORE_CONTRACT['identity']},
                'wind_contract': {**contract, 'bootstrap_online_recipe': GAUSSIAN_APEX_RECIPE,
                                  'online_recipe_migration': migration}}
    if requested == GATE_MARGIN_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == GATE_MARGIN_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != LOW_LR_RECIPE
                or resume is None or digest_file(resume) != GATE_MARGIN_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (12, 4800, 2238840832, 656175)
                or payload.get('stage_converged', False)):
            raise ValueError('Gate-margin reward requires the pinned full R11 learner')
        migration = dict(source_sha256=GATE_MARGIN_SOURCE_SHA256, source_update=4800,
            source_true_step=2238840832, source_adam_t=656175,
            previous_recipe=LOW_LR_RECIPE_ID, new_recipe=GATE_MARGIN_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='quality score curves only; learner, original gates and stage anchor exact')
        stack.online_recipe_migration = migration
        return {**payload, 'env_cfg': {**payload['env_cfg'],
                'wind_recovery_quality_score_mode': GATE_MARGIN_SCORE_CONTRACT['identity']},
                'wind_contract': {**contract, 'bootstrap_online_recipe': GATE_MARGIN_RECIPE,
                                  'online_recipe_migration': migration}}
    if requested == LOW_LR_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == LOW_LR_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != PRECISION_RECIPE
                or resume is None or digest_file(resume) != LOW_LR_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (10, 4300, 2173304832, 640175)
                or payload.get('stage_converged', False)):
            raise ValueError('Low-LR migration requires the pinned full R10 learner')
        migration = dict(source_sha256=LOW_LR_SOURCE_SHA256, source_update=4300,
            source_true_step=2173304832, source_adam_t=640175,
            previous_recipe=PRECISION_RECIPE_ID, new_recipe=LOW_LR_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='learning rate1.5e-5->7.5e-6 only; learner and stage anchor exact')
        stack.online_recipe_migration = migration
        return {**payload, 'args': {**payload['args'], 'learning_rate': 7.5e-6},
                'wind_contract': {**contract, 'bootstrap_online_recipe': LOW_LR_RECIPE,
                                  'online_recipe_migration': migration}}
    if requested == PRECISION_RECIPE_ID:
        if contract.get('bootstrap_online_recipe') == PRECISION_RECIPE:
            return payload
        if (contract.get('bootstrap_online_recipe') != JOINT_RECIPE
                or resume is None or digest_file(resume) != PRECISION_SOURCE_SHA256
                or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                    int(payload['opt'].t)) != (9, 3700, 2094661632, 620975)
                or payload.get('stage_converged', False)):
            raise ValueError('Precision migration requires the pinned full R8 learner')
        migration = dict(source_sha256=PRECISION_SOURCE_SHA256, source_update=3700,
            source_true_step=2094661632, source_adam_t=620975,
            previous_recipe=JOINT_RECIPE_ID, new_recipe=PRECISION_RECIPE_ID,
            previous_migration=stack.online_recipe_migration,
            change='effective sigma cap .02->.01; anchor .02->.005 keeps mean pull per squared action unchanged')
        stack.online_recipe_migration = migration
        return {**payload, 'args': {**payload['args'], 'max_log_std': math.log(.01),
                                   'actor_anchor_kl_coef': .005},
                'wind_contract': {**contract, 'bootstrap_online_recipe': PRECISION_RECIPE,
                                  'online_recipe_migration': migration}}
    if (getattr(stack, 'requested_online_recipe', None) != JOINT_RECIPE_ID
            or contract.get('bootstrap_online_recipe') == JOINT_RECIPE):
        return payload
    if (contract.get('bootstrap_online_recipe') != BOUNDED_RECIPE
            or resume is None or digest_file(resume) != JOINT_SOURCE_SHA256
            or (payload.get('stage_index'), payload.get('global_update'), payload.get('step'),
                int(payload['opt'].t)) != (9, 2739, 1968701440, 590223)
            or payload.get('stage_converged', False)):
        raise ValueError('Joint reward migration requires the pinned full R7 learner')
    migration = dict(source_sha256=JOINT_SOURCE_SHA256, source_update=2739,
        source_true_step=1968701440, source_adam_t=590223,
        previous_recipe=BOUNDED_RECIPE_ID, new_recipe=JOINT_RECIPE_ID,
        change='quality aggregation only; learner and stage anchor exact')
    stack.online_recipe_migration = migration
    return {**payload, 'env_cfg': {**payload['env_cfg'], 'wind_recovery_quality_aggregation':'geometric'},
            'wind_contract': {**contract, 'bootstrap_online_recipe':JOINT_RECIPE,
                              'online_recipe_migration':migration}}


def configure_recipe(stack, payload):
    """Select an authenticated bootstrap recipe before validating its course."""
    from train_fixed_plane_ball_obs_noise import numeric_digest
    requested = getattr(stack, 'requested_online_recipe', None)
    saved = payload.get('wind_contract', {}).get('bootstrap_online_recipe')
    direct = bool(payload.get('projection_bootstrap'))
    if requested is not None and requested not in RECIPES:
        raise ValueError('Unknown recovery online recipe')
    if saved is not None and saved != RECIPES.get(saved.get('identity')):
        raise ValueError('Stored recovery online recipe changed')
    if requested and not direct and saved != RECIPES[requested]:
        raise ValueError('Cannot reinterpret an existing PPO run with a new online recipe')
    enabled = bool(saved or requested)
    stack.spec = stack.default_spec
    stack.bootstrap_online_recipe = dict(saved or RECIPES[requested]) if enabled else None
    stack.bootstrap_stage_anchor = None
    stack.bootstrap_anchor_stage = None
    if not enabled:
        return
    if not getattr(stack, 'actor_bootstrap', None) or stack.spec.version != 325:
        raise ValueError('Recovery online recipe requires the pinned V325 actor bootstrap')
    from fixed_plane_v325_wind import WindContinuation

    class RecoveryContinuation(WindContinuation):
        @property
        def course(self):
            lessons = list(super().course)
            lessons[9] = replace(lessons[9], pose_start_deg=20., pose_ramp_updates=0,
                                 pose_quality_start_scale=1., pose_time_driven=False)
            return tuple(lessons)

    stack.spec = RecoveryContinuation(**asdict(stack.default_spec))
    if direct:
        stack.bootstrap_stage_anchor = payload['params']
        stack.bootstrap_anchor_stage = 9
    else:
        anchor = payload.get('bootstrap_stage_anchor_params')
        if (anchor is None or numeric_digest(anchor) != payload.get('bootstrap_stage_anchor_digest')
                or payload.get('bootstrap_stage_anchor_stage_index') != payload.get('stage_index')):
            raise ValueError('Recovery stage anchor is missing or changed')
        for key in ('teacher_pi', 'wind_pose_bound_gate_v1', 'residual_action_scale'):
            if numeric_digest(anchor[key]) != numeric_digest(payload['params'][key]):
                raise ValueError('Recovery stage anchor uses a different actor contract')
        stack.bootstrap_stage_anchor = anchor
        stack.bootstrap_anchor_stage = payload['stage_index']


def enter_stage(stack, params, stage):
    """Renew only at an actual course transition, keeping resumed anchors exact."""
    if getattr(stack, 'bootstrap_online_recipe', None) and stack.bootstrap_anchor_stage != stage:
        stack.bootstrap_stage_anchor = params
        stack.bootstrap_anchor_stage = stage


def checkpoint_fields(stack):
    from train_fixed_plane_ball_obs_noise import numeric_digest
    if not getattr(stack, 'bootstrap_online_recipe', None):
        return {}
    return dict(bootstrap_stage_anchor_params=stack.bootstrap_stage_anchor,
        bootstrap_stage_anchor_digest=numeric_digest(stack.bootstrap_stage_anchor),
        bootstrap_stage_anchor_stage_index=stack.bootstrap_anchor_stage)
