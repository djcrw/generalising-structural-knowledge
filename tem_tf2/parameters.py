#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import numpy as np
from scipy.special import comb
import itertools
from model_utils import DotDict as Dd


def default_params(width=None, height=None, world_type=None, batch_size=None):
    params = Dd()
    # use graph mode or eager mode
    params.graph_mode = True
    # tf_range=True compiles graph quicker but is slower when running.
    # e.g. for seq_len=75, tf_range=True is 30% slower than tf_range=False, but 80s vs 1000s compilation time
    params.tf_range = False

    params.batch_size = 16 if not batch_size else batch_size
    # seq_len - we truncate BPTT to sequences of this length
    params.seq_len = 75  # 75  # 50
    params.max_states = 350

    # 'rectangle', 'hexagonal', 'family_tree', 'line_ti', 'wood2000', 'frank2000', 'grieves2016', 'sun2020', 'nieh2021'
    params.world_type = 'frank2000' if not world_type else world_type

    # ENVIRONMENT params
    params.n_envs = params.batch_size
    params.s_size = 45
    params.asyncrounous_envs = True
    params = get_env_params(params, width, height=height)
    params.use_reward = True

    # DATA / SAVE / SUMMARY params params

    # only save date from first X of batch
    params.n_envs_save = 6
    # num gradient updates between summaries
    params.sum_int = 100
    # num gradient updates between detailed accuracy summaries
    params.sum_int_inferences = 400
    # number of gradient steps between saving data
    params.save_interval = int(50000 / params.seq_len)
    # number of gradient steps between saving model
    params.save_model = 5 * params.save_interval

    # MODEL params
    params.infer_g_type = 'g_p'  # 'g'
    params.two_hot = True
    params.s_size_comp = 10

    # numbers of variables for each frequency
    params.n_grids_all = [30, 30, 24, 18, 18]
    params.grid2phase = 3
    if params.world_type == 'tank':
        params.n_grids_all = [10, 10, 8, 6]
        params.grid2phase = 1.0
    if params.world_type == 'grieves2016':
        params.n_grids_all = [20, 20, 16, 12, 12]
        params.grid2phase = 2.0
    params.n_phases_all = [int(n_grid / params.grid2phase) for n_grid in params.n_grids_all]
    params.tot_phases = sum(params.n_phases_all)
    params.n_freq = len(params.n_phases_all)
    params.g_size = sum(params.n_grids_all)
    params.n_place_all = [p * params.s_size_comp for p in params.n_phases_all]
    params.p_size = sum(params.n_place_all)
    params.s_size_comp_hidden = 20 * params.s_size_comp
    params.prediction_freq = 0

    # These are smoothing parameters, so 'inverse frequencies': higher params.freqs = more smoothing = lower frequency
    params.freqs = sorted([0.01, 0.7, 0.91, 0.97, 0.99, 0.9995])[:params.n_freq]
    params.smooth_only_on_movement = False

    # initialisations
    params.g_init = 0.5
    params.p2g_init = 0.1
    params.g2g_init = 0.2

    # Build matrix for converting one-hot sensory data to two-hot: one_hot * two_hot_mat = two_hot
    params.two_hot_mat = onehot2twohot(np.expand_dims(np.eye(params.s_size), axis=0),
                                       combins_table(params.s_size_comp, 2), params.s_size_comp)

    # TRAINING params
    params.train_iters = 2000000
    params.train_on_visited_states_only = True
    params.learning_rate_max = 9.4e-4
    params.learning_rate_min = 8e-5
    params.logsig_ratio = 6
    params.logsig_offset = -2

    # losses
    params.which_costs = ['lx_p', 'lx_g', 'lx_gt', 'lp', 'lg', 'lg_reg', 'lp_reg']
    if 'p' in params.infer_g_type:
        params.which_costs.append('lp_x')

    # regularisation values
    params.g_reg_pen = 0.01
    params.p_reg_pen = 0.02
    params.weight_reg_val = 0.001

    # Number gradient updates for annealing (in number of gradient updates)
    params.temp_it = 2000
    params.forget_it = 200
    params.hebb_learn_it = 16000
    params.p2g_use_it = 0
    params.p2g_scale = 200
    params.p2g_sig_val = 10000
    params.g_reg_it = 40000000
    params.p_reg_it = 4000
    params.l_r_decay_steps = 4000
    params.l_r_decay_rate = 0.5

    # HEBB
    params.hebb_lim = 1
    params.lambd = 0.9999  # 1   #0.9999  # 1  # 0.9999
    params.eta = 0.5
    params.hebb_type = [[2], [2]]
    if 'p' not in params.infer_g_type:
        params.hebb_type = [2]

    # Build connectivity matrices for Hebbian weights, which is a list of modules TO, each a list of modules FROM:
    # If connectivity[x][y] is True, that means there is a connection FROM y TO x

    # R_f_F: hierarchical connections within grid modules
    params.R_f_F = connectivity_matrix(conn_hierarchical, params.freqs)
    # R_f_F_inv: all2all connections between and within grid modules
    params.R_f_F_inv = connectivity_matrix(conn_all2all, params.freqs)

    # In the model the place cell mask has the opposite (transpose) meaning from the connectivity matrices
    params.mask_p = get_mask(params.n_place_all, params.n_place_all, transpose_connectivity(params.R_f_F))

    # PLACE ATTRACTOR
    params.kappa = 0.8
    # Overall maximum number of attractor iterations
    params.n_recurs = params.n_freq
    # Set the number of maximum attractor iterations for each module to implement early stopping
    params.max_attractor_its = [params.n_recurs - f for f in range(params.n_freq)]
    params.max_attractor_its_inv = [params.n_recurs for _ in range(params.n_freq)]

    # STATE TRANSITION
    # R_G_F_f: hierarchical connections within grid modules
    params.R_G_F_f = connectivity_matrix(conn_hierarchical, params.freqs) if params.world_type != 'tank' else \
        connectivity_matrix(conn_separate, params.freqs)
    params.mask_g = get_mask(params.n_grids_all, params.n_grids_all, params.R_G_F_f)
    params.d_mixed_size = 15 if params.world_type == 'square' else 20

    # CHANGES SINCE NOW USING AS LITTLE CONTROL FLOW IN GRAPH AS POSSIBLE.
    # The following takes the existing connectivity setup but reorganises it, to optimise for tensorflow performance

    # List of attractor steps, each a list of freqs (in order) that want to update for each attractor step.
    params.attractor_freq_iterations = [[f for f in range(params.n_freq) if r < params.max_attractor_its[f]] for r in
                                        range(params.n_recurs)]
    params.attractor_freq_iterations_inv = [[f for f in range(params.n_freq) if r < params.max_attractor_its_inv[f]] for
                                            r in range(params.n_recurs)]

    # Here each R_f_F is a list (F) of lists (f), with number of frequency 'f' that influence frequency 'F'
    params.R_f_F_ = [list(np.where(x[:params.n_freq])[0]) for x in params.R_f_F]
    params.R_f_F_inv_ = [list(np.where(x[:params.n_freq])[0]) for x in params.R_f_F_inv]

    return params


def get_env_params(par, width, height):
    if par.world_type == 'rectangle':
        par_env = Dd({'stay_still': True,
                      'bias_type': 'angle',
                      'direc_bias': 0.25,
                      'angle_bias_change': 0.4,
                      'restart_max': 40,
                      'restart_min': 5,
                      'seq_jitter': 30,
                      'save_walk': 30,
                      'sum_inf_walk': 30,
                      'widths': [10, 10, 11, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9] if not width else
                      [width] * par.batch_size,
                      'heights': [10, 10, 11, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9] if not height else
                      [height] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right', 'stay still'],
                      })

    elif par.world_type == 'hexagonal':
        par_env = Dd({'stay_still': True,
                      'bias_type': 'angle',
                      'direc_bias': 0.2,
                      'angle_bias_change': 0.4,
                      'hex_boundary': True,
                      'restart_max': 40,
                      'restart_min': 5,
                      'seq_jitter': 30,
                      'save_walk': 30,
                      'sum_inf_walk': 30,
                      'widths': [6, 6, 7, 7, 5, 5, 6, 7, 5, 6, 6, 7, 5, 5, 6, 6] if not width else
                      [width] * par.batch_size,
                      'rels': ['down left', 'down right', 'up left', 'up right', 'left', 'right', 'stay still'],
                      })
        par_env.widths = [2 * x - 1 for x in par_env.widths]

    elif par.world_type == 'family_tree':
        par_env = Dd({'restart_max': 30,
                      'restart_min': 10,
                      'seq_jitter': 10,
                      'save_walk': 30,
                      'sum_inf_walk': 30,
                      'widths': [4, 4, 5, 5, 3, 3, 4, 4, 3, 5, 5, 4, 3, 4, 3, 5] if not width else
                      [width] * par.batch_size,
                      'rels': ['parent', 'child 1', 'child 2', 'sibling', 'grand parent', 'uncle/aunt',
                               'niece/nephew 1', 'niece/nephew 2', 'cousin 1', 'cousin 2'],
                      })

    elif par.world_type == 'line_ti':
        par_env = Dd({'jump_length': 4,
                      'restart_max': 30,
                      'restart_min': 10,
                      'seq_jitter': 10,
                      'save_walk': 30,
                      'sum_inf_walk': 30,
                      'widths': [5, 5, 6, 6, 4, 4, 5, 6, 4, 4, 5, 6, 4, 5, 6, 5] if not width else
                      [width] * par.batch_size,
                      'rels': ['two', 'components'],
                      })

    elif par.world_type == 'wood2000':
        par_env = Dd({'error_prob': 0.15,
                      'restart_max': 40,
                      'restart_min': 10,
                      'seq_jitter': 30,
                      'save_walk': 40,
                      'sum_inf_walk': 40,
                      'heights': [6 if width is None else height] * par.batch_size,
                      'widths': [4 if width is None else width] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right'],
                      })

    elif par.world_type == 'frank2000':
        par_env = Dd({'error_prob': 0.1,
                      'restart_max': 40,
                      'restart_min': 10,
                      'seq_jitter': 30,
                      'save_walk': 40,
                      'sum_inf_walk': 40,
                      'heights': [5 if width is None else height] * par.batch_size,
                      'widths': [1 if width is None else width] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right'],
                      })

    elif par.world_type == 'grieves2016':
        par_env = Dd({'error_prob': 0.1,
                      'switch_prob': 0.2,
                      'exploration_bias': 0.80,
                      'restart_max': 30,
                      'restart_min': 10,
                      'seq_jitter': 30,
                      'save_walk': 30,
                      'sum_inf_walk': 30,
                      'widths': [4 if width is None else width] * par.batch_size,
                      'simplified': True,
                      'rels': ['down', 'up', 'up-left', 'down-right', 'down-left', 'up-right', 'reward', 'no-reward',
                               'try reward']
                      })
        par_env.rels += ['0', '1', '2', '3'] if par_env.simplified else []

    elif par.world_type == 'sun2020':
        par_env = Dd({'restart_max': 40,
                      'restart_min': 10,
                      'seq_jitter': 30,
                      'save_walk': 40,
                      'sum_inf_walk': 40,
                      'widths': [3 if width is None else width] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right'],
                      'n_laps': 4,
                      })

    elif par.world_type == 'nieh2021':
        par_env = Dd({'error_beta': 0.5,
                      'bias': 4.5,
                      'restart_max': 10,
                      'restart_min': 2,
                      'seq_jitter': 4,
                      'save_walk': 10,
                      'sum_inf_walk': 10,
                      'widths': [8, 8, 10, 10, 12, 12, 6, 6, 12, 12, 10, 10, 8, 8, 6, 6] if not width else
                      [width] * par.batch_size,
                      'rels': ['proceed', 'pillar_left', 'pillar_right', 'left', 'right'],
                      })

    else:
        raise ValueError('incorrect world specified')

    par_env.n_actions = len(par_env.rels) if 'stay still' not in par_env.rels else len(par_env.rels) - 1
    par.n_actions = par_env.n_actions
    par.env = par_env
    return par


def get_n_states(par, width):
    # BEHAVIOUR / TASK params
    par.bias_type = 'angle'  # 'angle' or None. Bias to move in straight lines or no bias.
    par.direc_bias = 0.2 if par.world_type == 'hex' else 0.25  # strength of that bias
    par.angle_bias_change = 0.4  # random walk of angle rate
    par.stay_still = True
    par.hex_boundary = True  # give hexagonal boundary to hex worlds

    # latent state tasks
    par.splitter_error_prob = 0.15
    par.in_out_bound_error_prob = 0.15
    par.splitter_grieves_error_prob = 0.1
    par.splitter_grieves_switch_prob = 0.2
    par.splitter_grieves_exploration_bias = 0.6
    par.tank_beta = 0.5
    par.tank_bias = 4.5

    world_type, n_envs = par.world_type, par.n_envs
    poss_heights = [8, 8, 9, 9, 11, 11, 12, 12, 8, 8, 9, 9, 11, 11, 12, 12]
    reward_pos = [[0]] * par.batch_size
    no_reward_pos = [[0]] * par.batch_size
    reward_value = np.ones(par.batch_size)
    # those are [0] just se we don't divide by zero on par.reward_value = n_states[0] / len(reward_pos)
    n_laps = 4

    if world_type == 'hex':
        poss_widths = [6, 6, 7, 7, 5, 5, 6, 7, 5, 6, 6, 7, 5, 5, 6, 6] if not width else [width] * par.batch_size
        poss_widths = [2 * x - 1 for x in poss_widths]
        n_states = [(3 * (x ** 2) + 1) / 4 for x in poss_widths]
        rels = ['down left', 'down right', 'up left', 'up right', 'left', 'right', 'stay still']
        restart_max, restart_min, seq_jitter, save_walk, sum_inf_walk = 40, 5, 30, 30, 30

    elif world_type == 'square':
        poss_widths = [10, 10, 11, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9] \
            if not width else [width] * par.batch_size
        n_states = [x ** 2 for x in poss_widths]
        rels = ['down', 'up', 'left', 'right', 'stay still']
        restart_max, restart_min, seq_jitter, save_walk, sum_inf_walk = 40, 5, 30, 30, 30

    elif world_type == 'rectangle':
        poss_widths = [11, 11, 12, 12, 8, 8, 9, 9, 11, 11, 12, 12, 8, 8, 9, 9] \
            if not width else [width] * par.batch_size
        poss_heights = [8, 8, 9, 9, 11, 11, 12, 12, 8, 8, 9, 9, 11, 11, 12, 12] \
            if not width else [width] * par.batch_size
        n_states = [x * y for x, y in zip(poss_widths, poss_heights)]
        rels = ['down', 'up', 'left', 'right', 'stay still']
        restart_max, restart_min, seq_jitter, save_walk, sum_inf_walk = 40, 5, 30, 30, 30

    elif world_type == 'family_tree':
        poss_widths = [4, 4, 5, 5, 3, 3, 4, 4, 3, 5, 5, 4, 3, 4, 3, 5] if not width else [width] * par.batch_size
        n_states = [2 ** (x + 1) - 1 for x in poss_widths]
        rels = ['parent', 'child 1', 'child 2', 'sibling', 'grand parent', 'uncle/aunt', 'niece/nephew 1',
                'niece/nephew 2', 'cousin 1', 'cousin 2']
        restart_max, restart_min, seq_jitter, save_walk, sum_inf_walk = 30, 10, 10, 30, 30

    elif world_type == 'line_ti':
        poss_widths = [5, 5, 6, 6, 4, 4, 5, 6, 4, 4, 5, 6, 4, 5, 6, 5] if not width else [width] * par.batch_size
        n_states = [x for x in poss_widths]
        rels = ['two', 'components']  # here the actions are given in two components 1st is more/less, 2nd is how much
        restart_max, restart_min, seq_jitter, save_walk, sum_inf_walk = 30, 10, 10, 30, 30

    elif world_type == 'loop_laps':
        wid = 3
        poss_widths = [wid] * par.batch_size if not width else [width] * par.batch_size
        n_states = [n_laps * (2 * x + 2 * (x - 2)) for x in poss_widths]
        rels = ['down', 'up', 'left', 'right', 'stay still']
        reward_value = np.array(n_states)
        restart_max, restart_min, seq_jitter, save_walk, sum_inf_walk = 30, 10, 10, 30, 30

    elif world_type == 'splitter':
        # SORT THIS OUT
        wid = 4 if width is None else width
        hig = 6 if width is None else width
        poss_heights = [hig] * par.batch_size
        poss_widths = [wid] * par.batch_size
        n_states = [4 * (h - 1 + w - 1) + 2 * (w - 1) for h, w in zip(poss_heights, poss_widths)]
        reward_pos = [[h - 1 + w - 1, 3 * (h - 1 + w - 1)] for h, w in zip(poss_heights, poss_widths)]
        no_reward_pos = [[4 * (h - 1 + w - 1) + (w - 1) - 1, 4 * (h - 1 + w - 1) + 2 * (w - 1) - 1] for h, w in
                         zip(poss_heights, poss_widths)]
        reward_value = np.array(n_states) / 2
        rels = ['down', 'up', 'left', 'right']
        restart_max, restart_min, seq_jitter, save_walk, sum_inf_walk = 40, 10, 30, 40, 40

    elif world_type == 'in_out_bound':
        wid = 1 if width is None else width
        hig = 5 if width is None else width
        poss_heights = [hig] * par.batch_size
        poss_widths = [wid] * par.batch_size
        n_states = [6 * h + 6 * w + 4 * (h - 1) for h, w in zip(poss_heights, poss_widths)]
        reward_pos = [[0, 2 * h + w - 1, 3 * h + 2 * w + 1 * (h - 1) - 1, 4 * h + 3 * w + 2 * (h - 1) - 1] for h, w in
                      zip(poss_heights, poss_widths)]
        no_reward_pos = [[5 * h + 5 * w + 4 * (h - 1) - 1, 6 * h + 6 * w + 4 * (h - 1) - 1] for h, w in
                         zip(poss_heights, poss_widths)]
        reward_value = np.array(n_states) / 2
        rels = ['down', 'up', 'left', 'right']
        restart_max, restart_min, seq_jitter, save_walk, sum_inf_walk = 40, 10, 30, 40, 40

    elif world_type == 'splitter_grieves':
        seg_len = 4 if width is None else width
        poss_widths = [seg_len] * par.batch_size
        map_states = 7 * seg_len + 8 + 4 + 1
        n_states = [8 * map_states for _ in poss_widths]

        # THIS NEEDS FIXING FOR REWARD POS BEING BATCH DEPENDENT
        # choices: middle, left, right
        choice_points = [1 * seg_len + 1, 2 * seg_len + 2, 5 * seg_len + 7]
        # goal boxes: left, center-left, center-right, box right
        goal_boxes = [3 * seg_len + 3 + 0, 4 * seg_len + 4 + 1, 6 * seg_len + 6 + 2, 7 * seg_len + 7 + 3]

        reward_pos = [(a + 1) * map_states - 1 for a in [0, 1, 2, 3]] + [b * map_states + a + 1 for a, b in
                                                                         zip(goal_boxes, [4, 5, 6, 7])]
        reward_pos.sort()
        no_reward_pos = [b * map_states + a + 1 for a, b in itertools.product(goal_boxes, [0, 1, 2, 3, 4, 5, 6, 7]) if
                         b * map_states + a + 1 not in reward_pos]
        no_reward_pos.sort()
        reward_pos = [reward_pos] * par.batch_size
        no_reward_pos = [no_reward_pos] * par.batch_size
        reward_value = np.array(n_states) / 4
        rels = ['down', 'up', 'up-left', 'down-right', 'down-left', 'up-right', 'reward', 'no-reward', 'try reward']
        par.map_states = map_states
        par.choice_points = choice_points
        par.goal_boxes = goal_boxes

        par.grieves_reward_pos_training = reward_pos[0][:4] + [x for x in no_reward_pos[0] if
                                                               x < 4 * map_states and x % map_states != goal_boxes[
                                                                   np.floor(x / map_states).astype(int)] + 1]
        restart_max, restart_min, seq_jitter, save_walk, sum_inf_walk = 40, 10, 30, 40, 40

    elif world_type == 'tank':
        wid = width if width is None else width
        poss_widths = [8, 8, 10, 10, 12, 12, 6, 6, 12, 12, 10, 10, 8, 8, 6, 6] if not width else [wid] * par.batch_size
        n_states = [x ** 2 + 4 for x in poss_widths]
        reward_pos = [[x - 1, x - 4] for x in n_states]
        no_reward_pos = [[x - 2, x - 3] for x in n_states]
        reward_value = np.array(poss_widths)
        rels = ['proceed', 'pillar_left', 'pillar_right', 'left', 'right']
        restart_max, restart_min, seq_jitter, save_walk, sum_inf_walk = 10, 2, 4, 10, 10

    else:
        raise ValueError('incorrect world specified')

    poss_widths = list(np.tile(poss_widths, int(np.ceil(n_envs / len(poss_widths))))[:n_envs])
    poss_heights = list(np.tile(poss_heights, int(np.ceil(n_envs / len(poss_heights))))[:n_envs])

    # set limits on number of steps in a environment (in terms of gradient updates - hence divide by seq_len)
    # decay mean walk length from from restart_max to restart_min. Sample random int with limits +- seq_jitter. So
    # actual walk limits are mean +- random_int. Want to sample in proportion to 1/length seq.
    par.restart_max = restart_max
    par.restart_min = restart_min
    par.seq_jitter = seq_jitter

    # length of walk for saving
    par.save_walk = save_walk
    # length of walk for summary inferences
    par.sum_inf_walk = sum_inf_walk

    par.n_laps = n_laps
    par.reward_pos = reward_pos
    par.no_reward_pos = no_reward_pos
    par.reward_value = reward_value.astype(np.float32)  # make same as predicting other sensory experiences

    par.n_actions = len(rels) if 'stay still' not in rels else len(rels) - 1
    par.widths = poss_widths
    par.n_states = [int(x) for x in n_states]
    # hex world gets trimmed down from square to hexagon
    par.n_states_world = par.n_states if 'hex' not in world_type else [x ** 2 for x in poss_widths]
    par.heights = poss_heights
    par.relationship_types = rels

    par.restart_max, par.restart_min, par.seq_jitter = restart_max, restart_min, seq_jitter

    return par


def get_scaling_parameters(index, par):
    # these scale with number of gradient updates
    temp = np.minimum((index + 1) / par.temp_it, 1.0)
    forget = np.minimum((index + 1) / par.forget_it, 1.0)
    hebb_learn = np.minimum((index + 1) / par.hebb_learn_it, 1.0)
    p2g_use = sigmoid((index - par.p2g_use_it) / par.p2g_scale)
    l_r = (par.learning_rate_max - par.learning_rate_min) * (par.l_r_decay_rate ** (
            index / par.l_r_decay_steps)) + par.learning_rate_min
    l_r = np.maximum(l_r, par.learning_rate_min)
    g_cell_reg = 1 - np.minimum((index + 1) / par.g_reg_it, 1.0)
    p_cell_reg = 1 - np.minimum((index + 1) / par.p_reg_it, 1.0)

    scalings = Dd({'temp': temp,
                   'forget': forget,
                   'h_l': hebb_learn,
                   'p2g_use': p2g_use,
                   'l_r': l_r,
                   'g_cell_reg': g_cell_reg,
                   'p_cell_reg': p_cell_reg,
                   'iteration': index,
                   })

    return scalings


def connectivity_matrix(g2g, freqs):
    """
    Build connectivity matrices between modules. C is a list of modules TO, each a list of modules FROM:
    If C[x][y] is True, that means there is a connection FROM y TO x
    g2g are functions that return whether a connection exists, given the 'frequency'
    (actually, exponential smoothing - so more like inverse frequency) of both modules
    """
    connec = [[None for _ in range(len(freqs))] for _ in range(len(freqs))]
    for f_from in range(len(freqs)):
        for f_to in range(len(freqs)):
            connec[f_to][f_from] = g2g(freqs[f_from], freqs[f_to])
    return connec


def transpose_connectivity(connec):
    """
    C is a list of modules TO, each a list of modules FROM: if C[x][y] is True, that means there is a connection 
    FROM y TO x. This function calculates the transpose, collecting the ith entry of each input inner list 
    in the ith output inner list. Thus if C_T[x][y] is True, there is a connection FROM x TO y.
    """
    connec_t = [list(entry_i) for entry_i in zip(*connec)]
    return connec_t


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_mask(n_cells_in, n_cells_out, r):
    """
    Generate a mask matrix M_ij that for each cell i holds if it recieves input from cell (i.e. connection from j to i)
    Input a list of cells per module and a connectivity matrix r_ij, which is list of lists that indicates the
    connectivity from module j to i: if r[i][j] is True, then module i recieves input from module j
    """

    n_freq = len(n_cells_in)
    n_all_in = sum(n_cells_in)
    n_all_out = sum(n_cells_out)
    c_p_in = np.insert(np.cumsum(n_cells_in), 0, 0).astype(int)
    c_p_out = np.insert(np.cumsum(n_cells_out), 0, 0).astype(int)

    mask = np.zeros((n_all_in, n_all_out), dtype=np.float32)

    for f_to in range(n_freq):
        for f_from in range(n_freq):
            mask[c_p_in[f_to]:c_p_in[f_to + 1], c_p_out[f_from]:c_p_out[f_from + 1]] = r[f_to][f_from]

    return mask


def combins(n, k, m):
    s = []
    for i in range(1, n + 1):
        c = comb(n - i, k)
        if m >= c:
            s.append(1)
            m -= c
            k = k - 1
        else:
            s.append(0)
    return tuple(s)


def combins_table(n, k, map_max=None):
    table = []
    rev_table = {}
    table_top = comb(n, k)
    for m in range(int(table_top)):
        # forward mapping
        c = combins(n, k, m)
        if map_max is None or m < map_max:
            table.append(c)
            rev_table[c] = m
        else:
            rev_table[c] = m % map_max
    return table


def onehot2twohot(onehot, table, compress_size):
    seq_len = np.shape(onehot)[2]
    batch_size = np.shape(onehot)[0]
    twohot = np.zeros((batch_size, compress_size, seq_len))
    for i in range(np.shape(onehot)[2]):
        vals = np.argmax(onehot[:, :, i], 1)
        for b in range(np.shape(onehot)[0]):
            twohot[b, :, i] = table[vals[int(b)]]

    return twohot


# Specify types of connections between modules, from initial values of exponential smoothing a ('inverse frequency')
def conn_hierarchical(a_from, a_to):
    return int(a_from >= a_to)  # Allow connections only from low to high frequency


def conn_separate(a_from, a_to):
    return int(a_from == a_to)  # Allow connections only within frequency


def conn_all2all(*_):
    return int(True)  # Allow all connections, independent of frequencies of modules


def conn_none2none(*_):
    return int(False)  # Allow no connections at all, independent of frequencies


def old2new(world_type):
    old2new_name_convert = Dd({'hex': 'hexagonal',
                               'splitter': 'wood2000',
                               'in_out_bound': 'frank2000',
                               'splitter_grieves': 'grieves2016',
                               'loop_laps': 'sun2020',
                               'tank': 'nieh2021',
                               'rectangle': 'rectangle',
                               'square': 'rectangle',
                               'hexagonal': 'hexagonal',
                               'wood2000': 'wood2000',
                               'frank2000': 'frank2000',
                               'grieves2016': 'grieves2016',
                               'sun2020': 'sun2020',
                               'nieh2021': 'nieh2021',
                               })
    try:
        return old2new_name_convert[world_type]
    except KeyError:
        return world_type
