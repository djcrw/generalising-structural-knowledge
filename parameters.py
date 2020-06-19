#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

from arb_functions import *
from helper_functions import *
import numpy as np


def default_params():
    params = dict()

    params['batch_size'] = 16
    params['BPTT_truncation'] = 25  # 100 for loop_laps world

    # ENVIRONMENT PARAMS

    params['world_type'] = 'hex'  # 'hex', 'square', 'family_tree', 'line_ti', 'loop_laps', 'rectangle'
    params['hex_boundary'] = True  # give hexagonal boundary to hex worlds
    params['n_envs'] = params['batch_size']
    params['diff_env_batches_envs'] = np.arange(params['batch_size'])  # which batch in which environment

    # loop_laps world options
    params['n_laps'] = 4
    params['reward_pos'] = 0  # loop_laps world - where 'reward' is

    params['widths'], params['n_states'], params['n_states_world'], params['n_actions'], params['jump_length'], \
        params['heights'] = get_n_states(params)
    params['reward_value'] = params['n_states'][0]  # make same as predicting other sensory experiences

    # BEHAVIOUR PARAMS

    params['poss_behaviours'] = ['normal']  # '['shiny', 'normal'] for OVC cells, ['normal'] otherwise
    params['bias_type'] = 'angle'  # 'angle' or None. Bias to move in straight lines or no bias.
    params['direc_bias'] = 0.2  # strength of that bias
    params['angle_bias_change'] = 0.4  # random walk of angle rate
    params['stay_still'] = True

    # for ovc cell simulations
    params['shiny_bias'] = (0.2, 0.55)  # (general bias to current object, distance bias to object)
    params['object_stay_still'] = 0.05
    params['n_shiny_senses'] = 10
    params['n_shiny'] = [3, 4, 5]
    params['object_hang_min'] = 15
    params['object_hang_max'] = 20
    params['ovc_module_use'] = True  # option for a separate module that receives object info
    params['ovc_module_num'] = 0
    params['shiny_sense'] = 0

    # DATA / SAVE / SUMMARY PARAMS PARAMS

    params['seq_len'] = params['BPTT_truncation']
    params['s_size'] = 45
    params['restart_max'] = np.ceil(6000 / params['seq_len']).astype(int)
    params['restart_min'] = np.ceil(2500 / params['seq_len']).astype(int)
    params['seq_jitter'] = np.ceil(200 / params['seq_len']).astype(int)
    params['link_inf_walk'] = int(3000 / params['seq_len'])
    lim = int(10 * np.median(params['n_states_world']))
    lim_seq = np.ceil(lim / params['seq_len']).astype(int)
    if params['world_type'] in ['family_tree', 'line_ti']:
        params['restart_max'], params['restart_min'] = 2 * lim_seq, lim_seq
        params['seq_jitter'] = np.ceil(np.median(params['n_states_world']) / params['seq_len']).astype(int)
        params['link_inf_walk'] = params['restart_min']
    if params['world_type'] == 'loop_laps':
        params['restart_max'], params['restart_min'] = 2 * lim_seq, lim_seq
        params['seq_jitter'] = np.ceil(np.median(params['n_states_world']) / params['seq_len']).astype(int)
        params['link_inf_walk'] = params['restart_min']
    params['curriculum_steps'] = 12 / params['seq_len']  # number of steps dropped per environment switch
    params['n_envs_save'] = 6  # only save date from first X of batch
    params['sum_int'] = 200
    params['n_save_data'] = int(25 * max(params['n_states']) / params['seq_len'])
    params['save_interval'] = int(int(50000 / params['seq_len']) / params['n_save_data']) * params['n_save_data']
    params['n_walk'] = params['seq_len']
    params['n_envs_test'] = np.ceil(params['n_envs'] / 2)
    params['save_model'] = 5 * params['save_interval']

    # MODEL PARAMS

    params['infer_g_type'] = 'g_p'  # 'g'
    if 'shiny' in params['poss_behaviours']:
        params['infer_g_type'] += '_x'

    params['two_hot'] = True
    params['s_size_comp'] = 10
    params['no_direc_gen'] = True if 'shiny' in params['poss_behaviours'] else False

    # numbers of variables for each frequency
    n_phases_all = [10, 10, 8, 6, 6]
    if params['world_type'] in ['loop_laps']:
        n_phases_all = [6, 6, 5, 4, 4]
    if 'shiny' in params['poss_behaviours']:
        if params['ovc_module_use']:
            # this is a separate factorised module for object vector cells
            # this module will be '2nd' in hierarchy - so make sure rest of params know this.
            if len(n_phases_all) > 4:
                n_phases_all[params['ovc_module_num']] = 6
            else:
                n_phases_all.insert(params['ovc_module_num'], 6)
        else:
            n_phases_all = [x + 2 for x in n_phases_all]
    params['n_phases_all'] = n_phases_all
    params['n_place_all'] = [p * params['s_size_comp'] for p in params['n_phases_all']]
    params['n_grids_all'] = [int(3 * n_phase) for n_phase in params['n_phases_all']]
    params['tot_phases'] = sum(params['n_phases_all'])
    params['n_freq'] = len(params['n_phases_all'])
    params['g_size'] = sum(params['n_grids_all'])
    params['p_size'] = int(params['tot_phases'] * params['s_size_comp'])
    params['s_size_comp_hidden'] = 20 * params['s_size_comp']
    params['prediction_freq'] = 0 if not ('shiny' in params['poss_behaviours'] and params['ovc_module_num'] == 0) else 1
    params['n_senses'] = [params['s_size']] * params['n_freq']

    params['freqs'] = [0.01, 0.7, 0.91, 0.97, 0.99, 0.9995]
    if 'shiny' in params['poss_behaviours'] and params['ovc_module_use']:
        params['freqs'].insert(params['ovc_module_num'], 0.01)

    # initialisations
    params['g_init'] = 0.5
    params['p2g_init'] = 0.1
    params['x2g_init'] = 0.2

    # activations
    params['p_activation'] = lambda x: tf.nn.leaky_relu(tf.minimum(tf.maximum(x, -1), 1), alpha=0.01)
    params['g2g_activation'] = lambda x: tf.minimum(tf.maximum(x, -1), 1)
    params['ovc_activation'] = lambda x: tf.nn.leaky_relu(tf.minimum(tf.maximum(x, -1), 1), alpha=0.01)

    # TRAINING PARAMS
    params['train_iters'] = int(10000)
    params['optimiser'] = 'Adam'
    params['train_on_visited_states_only'] = True
    params['learning_rate_max'] = 9.4e-4
    params['learning_rate_min'] = 8e-5
    params['train_sig_p2g'] = True if 'p' in params['infer_g_type'] else False
    params['train_sig_g2g'] = True if 'g' in params['infer_g_type'] else False
    params['train_sig_g2g_i'] = True if 'g' in params['infer_g_type'] else False
    params['train_sig_x2g'] = True if 'x' in params['infer_g_type'] else False
    params['logsig_ratio'] = 6
    params['logsig_offset'] = -2

    # losses
    params['which_costs'] = ['lx_p', 'lx_g', 'lx_gt', 'lp', 'lg', 'lg_reg', 'lp_reg']
    if 'p' in params['infer_g_type']:
        params['which_costs'].append('lp_x')
    if 'shiny' in params['poss_behaviours']:
        params['which_costs'].append('ovc_reg')
        params['which_costs'].append('weight_reg')

    # regularisation values
    params['g_reg_pen'] = 0.01
    params['p_reg_pen'] = 0.02
    params['ovc_reg_pen'] = 0.02
    params['weight_reg_val'] = 0.001

    # Number gradient updates for annealing
    params['temp_it'] = 2000
    params['forget_it'] = 200
    params['hebb_learn_it'] = 16000
    params['p2g_use_it'] = 400
    params['p2g_scale'] = 200
    params['p2g_sig_val'] = 10000
    params['ovc_reg_it'] = 4000
    params['g_reg_it'] = 40000000
    params['p_reg_it'] = 4000
    params['l_r_decay_steps'] = 4000
    params['l_r_decay_rate'] = 0.5

    # HEBB
    params['hebb_mat_max'] = 1
    params['lambd'] = 0.9999
    params['eta'] = 0.5
    params['hebb_type'] = [[2], [2]]
    if 'p' not in params['infer_g_type']:
        params['hebb_type'] = [2]

    # Types of allowed connections in Hebbian matrices
    hierarchical = [[1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 1]]

    hierarchical_t = [[1, 0, 0, 0, 0, 0],
                      [1, 1, 0, 0, 0, 0],
                      [1, 1, 1, 0, 0, 0],
                      [1, 1, 1, 1, 0, 0],
                      [1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 1, 1]]

    separate = [[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]]

    all2all = [[1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1]]

    # R_f_F says how frequency f influences frequency F
    params['R_f_F'] = cp.deepcopy(hierarchical_t)
    params['R_f_F_inv'] = cp.deepcopy(all2all)

    if 'shiny' in params['poss_behaviours'] and params['ovc_module_use']:
        # OVC projects to and receives from all
        params['R_f_F'] = np.asarray(params['R_f_F'])
        params['R_f_F'][params['ovc_module_num'], :] = [1, 1, 1, 1, 1, 1]  # OVC projects to
        params['R_f_F'][:, params['ovc_module_num']] = [1, 1, 1, 1, 1, 1]  # OVC receives
        params['R_f_F'] = params['R_f_F'].tolist()
    params['mask_p'] = place_mask(params['n_phases_all'], params['s_size_comp'], params['R_f_F'])

    # PLACE ATTRACTOR
    params['which_way'] = ['normal', 'normal'] if len(params['hebb_type']) < 2 else ['normal', 'inv']
    params['prev_p_decay'] = 0.8
    params['n_recurs'] = params['n_freq'] - 1 if 'shiny' in params['poss_behaviours'] and params['ovc_module_use'] \
        else params['n_freq']
    params['Hebb_diff_freq_its_max'] = [params['n_recurs'] - freq for freq in range(params['n_recurs'])]
    params['Hebb_inv_diff_freq_its_max'] = [params['n_recurs'] for _ in range(params['n_recurs'])]
    if 'shiny' in params['poss_behaviours'] and params['ovc_module_use']:
        params['Hebb_diff_freq_its_max'] = [params['Hebb_diff_freq_its_max'][0]] + params['Hebb_diff_freq_its_max']
        params['Hebb_inv_diff_freq_its_max'] = [params['Hebb_inv_diff_freq_its_max'][0]] + params[
            'Hebb_inv_diff_freq_its_max']

    # STATE TRANSITION
    # R_G_F_f says how frequency f influences frequency F (opposite to R_F_f_F)
    params['R_G_F_f'] = cp.deepcopy(separate) if 'shiny' in params['poss_behaviours'] else cp.deepcopy(hierarchical)

    if 'shiny' in params['poss_behaviours'] and params['ovc_module_use']:
        # OVC acts alone
        params['R_G_F_f'] = np.asarray(params['R_G_F_f'])
        params['R_G_F_f'][params['ovc_module_num'], :] = [1 if i == params['ovc_module_num'] else 0 for i in
                                                          range(len(params['R_G_F_f']))]
        params['R_G_F_f'][:, params['ovc_module_num']] = [1 if i == params['ovc_module_num'] else 0 for i in
                                                          range(len(params['R_G_F_f']))]
        params['R_G_F_f'] = params['R_G_F_f'].tolist()

    params['mask_g'] = grid_mask(params['n_grids_all'], params['R_G_F_f'])
    params['d_mixed'] = True
    params['d_mixed_size'] = 15 if params['world_type'] == 'square' else 20

    params['x2g_freqs'] = [False, False, False, False, False, False]
    if params['ovc_module_use'] and 'shiny' in params['poss_behaviours']:
        params['x2g_freqs'][params['ovc_module_num']] = True
    else:
        params['x2g_freqs'] = [True for _ in params['x2g_freqs']]

    return params


def get_n_states(pars):
    world_type, n_envs, n_laps = pars['world_type'], pars['n_envs'], pars['n_laps']
    poss_heights = [8, 8, 9, 9, 11, 11, 12, 12, 8, 8, 9, 9, 11, 11, 12, 12]

    if world_type == 'hex':
        poss_widths = [6, 6, 7, 7, 5, 5, 6, 7, 5, 6, 6, 7, 5, 5, 6, 6]
        poss_widths = [2 * x - 1 for x in poss_widths]
        n_states = [(3 * (x ** 2) + 1) / 4 for x in poss_widths]
        n_actions = 6

    elif world_type == 'square':
        poss_widths = [10, 10, 11, 11, 8, 9, 10, 11, 8, 9, 10, 11, 8, 8, 9, 9]
        n_states = [x ** 2 for x in poss_widths]
        n_actions = 4

    elif world_type == 'rectangle':
        poss_widths = [11, 11, 12, 12, 8, 8, 9, 9, 11, 11, 12, 12, 8, 8, 9, 9]
        poss_heights = [8, 8, 9, 9, 11, 11, 12, 12, 8, 8, 9, 9, 11, 11, 12, 12]
        n_states = [x * y for x, y in zip(poss_widths, poss_heights)]
        n_actions = 4

    elif world_type == 'family_tree':
        poss_widths = [4, 4, 5, 5, 3, 3, 4, 4, 3, 5, 5, 4, 3, 4, 3, 5]
        n_states = [2 ** (x + 1) - 1 for x in poss_widths]
        n_actions = 10

    elif world_type == 'line_ti':
        poss_widths = [5, 5, 6, 6, 4, 4, 5, 6, 4, 4, 5, 6, 4, 5, 6, 5]
        n_states = [x for x in poss_widths]
        n_actions = 2

    elif world_type == 'loop_laps':
        poss_widths = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        poss_widths = [x + -1 for x in poss_widths]
        n_states = [n_laps * (2 * x + 2 * (x - 2)) for x in poss_widths]
        n_actions = 4

    else:
        n_states = None
        n_actions = None
        poss_widths = None
        poss_heights = None

    poss_widths = list(np.tile(poss_widths, int(np.ceil(n_envs / len(poss_widths))))[:n_envs])
    poss_heights = list(np.tile(poss_heights, int(np.ceil(n_envs / len(poss_heights))))[:n_envs])

    if world_type == 'hex':
        n_states_world = [x ** 2 for x in poss_widths]

    elif world_type == 'square':
        n_states_world = [x ** 2 for x in poss_widths]

    elif world_type == 'rectangle':
        n_states_world = [x * y for x, y in zip(poss_widths, poss_heights)]

    elif world_type == 'family_tree':
        n_states_world = [2 ** (x + 1) - 1 for x in poss_widths]

    elif world_type == 'line_ti':
        n_states_world = [x for x in poss_widths]

    elif world_type == 'loop_laps':
        n_states_world = [n_laps * (2 * x + 2 * (x - 2)) for x in poss_widths]

    else:
        n_states_world = None

    jump_length = [x - 2 for x in poss_widths]

    return poss_widths, n_states, n_states_world, n_actions, jump_length, poss_heights
