#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

from environment_functions import *
import numpy as np
import copy as cp
import os
import datetime
import logging
from distutils.dir_util import copy_tree
import scipy
import inspect


def cell_norm_online(cells, positions, current_cell_mat, pars):
    # for separate environments within each batch
    envs = pars['diff_env_batches_envs']
    n_states = pars['n_states_world']
    n_envs_save = pars['n_envs_save']

    num_cells = np.shape(cells)[1]
    n_trials = np.shape(cells)[2]

    cell_mat = [np.zeros((n_states[envs[env]], num_cells)) for env in range(n_envs_save)]

    new_cell_mat = [None] * n_envs_save

    for env in range(n_envs_save):
        for ii in range(n_trials):
            position = int(positions[env, ii])
            cell_mat[env][position, :] += cells[env, :, ii]
        try:
            new_cell_mat[env] = cell_mat[env] + current_cell_mat[env]
        except:
            new_cell_mat[env] = cell_mat[env]

    return new_cell_mat


def accuracy_positions_online(cells, positions, current_cell_mat, pars):
    # for separate environments within each batch
    envs = pars['diff_env_batches_envs']
    n_states = pars['n_states_world']
    n_envs_save = pars['n_envs_save']
    n_trials = np.shape(cells)[1]

    cell_mat = [np.zeros(n_states[envs[env]]) for env in range(n_envs_save)]

    new_cell_mat = [None] * n_envs_save

    for env in range(n_envs_save):
        for ii in range(n_trials):
            position = int(positions[env, ii])
            cell_mat[env][position] += cells[env, ii]

        try:
            new_cell_mat[env] = cell_mat[env] + current_cell_mat[env]
        except:
            new_cell_mat[env] = cell_mat[env]

    return new_cell_mat


def correct_action_pred_np(real, pred):
    correct_prediction = np.equal(real, np.round(pred)).astype(np.float32)
    accuracy = np.floor(np.mean(correct_prediction, 1, keepdims=True))
    return accuracy.astype(np.int32)


def positions_online(position, positions, n_envs_save):
    pos = position[:, 1:]  # doesn't include initial 'start state'
    for env in range(n_envs_save):
        try:
            positions[env] = np.concatenate((positions[env], pos[env]), axis=0)
        except:
            positions[env] = cp.deepcopy(pos[env])
    return positions


def sense_online(sense, senses, n_seqs):
    senses_ = []
    for seq in range(n_seqs):
        senses_.append(np.argmax(sense[seq], 1))
    senses_ = np.transpose(np.squeeze(np.array(senses_)), [1, 0])

    try:
        senses = np.concatenate((senses, senses_), axis=1)
    except:
        senses = cp.deepcopy(senses_)

    return senses


def accuracy_online(accs, acc_fn, real, pred, n_seqs):
    acc = []
    for seq in range(n_seqs):
        acc.append(acc_fn(real[:, :, seq], pred[seq]))
    acc = np.transpose(np.squeeze(np.array(acc)), [1, 0])

    try:
        accs = np.concatenate((accs, acc), axis=1)
    except:
        accs = cp.deepcopy(acc)

    return acc, accs


def acc_sense(real, pred):
    accuracy = np.equal(np.argmax(real, 1), np.argmax(pred, 1))
    accuracy = np.expand_dims(accuracy, 1)
    return accuracy.astype(np.int32)


def place_mask(n_phases, s_size, rff):
    # mask - only allow across freq, within sense connections
    # p_size : total place cell size
    # s_size : total number of senses
    # n_phases : number of phases in each frequency
    tot_phases = sum(n_phases)
    p_size = s_size * tot_phases
    cum_phases = np.cumsum(n_phases)

    c_p = np.insert(cum_phases*s_size, 0, 0).astype(int)

    mask = np.zeros((p_size, p_size), dtype=np.float32)

    for freq_row in range(len(n_phases)):
        for freq_col in range(len(n_phases)):
            mask[c_p[freq_row]:c_p[freq_row+1], c_p[freq_col]:c_p[freq_col+1]] = rff[freq_row][freq_col]

    return mask


def grid_mask(n_phases, r):
    g_size = sum(n_phases)
    cum_phases = np.cumsum(n_phases)

    c_p = np.insert(cum_phases, 0, 0).astype(int)

    mask = np.zeros((g_size, g_size), dtype=np.float32)

    for freq_row in range(len(n_phases)):
        for freq_col in range(len(n_phases)):
            mask[c_p[freq_row]:c_p[freq_row + 1], c_p[freq_col]:c_p[freq_col + 1]] = r[freq_row][freq_col]

    return mask


def make_directories():
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    run = 0
    dir_check = True
    train_path, model_path, save_path, script_path, gen_path = None, None, None, None, None
    while dir_check:
        gen_path = '../Summaries/' + date + '/run' + str(run) + '/'
        train_path = gen_path + 'train'
        model_path = gen_path + 'model'
        save_path = gen_path + 'save'
        script_path = gen_path + 'script'
        run += 1
        if not os.path.exists(train_path) and not os.path.exists(model_path) and not os.path.exists(save_path):
            os.makedirs(train_path)
            os.makedirs(model_path)
            os.makedirs(save_path)
            os.makedirs(script_path)
            dir_check = False

    return gen_path, train_path, model_path, save_path, script_path


def make_logger(gen_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(gen_path + 'report.log')
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)

    return logger


def save_params(pars, save_path, script_path):
    par_copy = cp.deepcopy(pars)
    for key, value in par_copy.items():
        if 'function' in str(value):
            fun_str = inspect.getsourcelines(value)[0][0]
            if 'params' in fun_str:
                par_copy[key] = fun_str.split('lambda ')[1]
            elif 'def' in fun_str:
                par_copy[key] = fun_str.split('def')[1]
            else:
                par_copy[key] = fun_str  # str(value).split(' at')[0].split('<')[1]
    np.save(save_path + '/params', par_copy)
    copy_tree('./', script_path)

    return


def get_next_batch(walking_data, x_data, states_mat, index, visited, pars):
    envs = pars['diff_env_batches_envs']

    position, direc = walking_data

    data = get_new_data_diff_envs(position, x_data, envs, states_mat, pars)

    xs = data[:, :, 1: pars['seq_len'] + 1]
    ds = direc[:, :, :pars['seq_len']]

    x_data, start_state = data, position[:, -1].astype(int)

    # position is n_seq + 1, where 1st element is 'start state position'
    if pars['train_on_visited_states_only']:
        s_visited = np.zeros((pars['batch_size'], pars['seq_len']))
        for b in range(pars['batch_size']):
            for seq in range(pars['seq_len']):
                pos = int(position[b, seq + 1])  # 'current' position (+1 as that's where we start)
                s_visited[b, seq] = 1 if visited[b, pos] == 1 else 0  # have I visited this position before
                visited[b, pos] = 1  # add position to places I've been
    else:
        s_visited = np.ones((pars['batch_size'], pars['seq_len']))

    new_data = (xs, ds, position, visited, s_visited)
    old_data = (x_data, start_state)

    temp = np.minimum(1, (index + 1) / pars['temp_it'])
    forget = np.minimum(1, (index + 1) / pars['forget_it'])
    hebb_learn = np.minimum(1, (index + 1) / pars['hebb_learn_it'])
    p2g_use = np.minimum(1, (index + 1) / pars['p2g_use_it'])

    model_vars = [temp, forget, hebb_learn, p2g_use]

    return new_data, old_data, model_vars


def initialise_variables(par, adjs):
    gs = np.maximum(np.random.randn(par['batch_size'], par['g_size']) * par['g_init'], 0)
    x_s = np.zeros((par['batch_size'], par['s_size_comp'] * par['n_freq']))
    x_data = np.zeros((par['batch_size'], par['s_size'], par['n_walk'] + 1))

    n_states = par['n_states_world']
    visited = np.zeros((par['batch_size'], max(n_states)))

    envs = par['diff_env_batches_envs']

    start_state = np.zeros(par['batch_size'])
    for batch in range(par['batch_size']):
        # this needs to be sorted out for hex not box worlds
        allowed_states = np.where(np.sum(adjs[envs[batch]], 1) > 0)[0]  # only include states you can get to
        if par['world_type'] in ['loop_laps']:
            start_state[batch] = 0
        else:
            start_state[batch] = np.random.choice(allowed_states)
    prev_direc = np.random.randint(0, par['n_actions']+1, par['batch_size']).astype(int)

    return gs, x_s, x_data, start_state, prev_direc, visited


def save_data_maps(positions, data_list, save_path, n_envs_save, index, states, names):
    pos_count = [0] * n_envs_save

    for env in range(n_envs_save):
        pos_count[env] = np.bincount(positions[env].astype(np.int32).flatten(), minlength=states[env]) + 0.001
    np.save(save_path + '/pos_count_' + str(index), pos_count)

    for data, name in zip(data_list, names):
        data_map = [None] * n_envs_save
        for env in range(n_envs_save):
            try:
                data_map[env] = np.matmul(np.diag(1 / pos_count[env]), data[env])
            except:
                pass

        np.save(save_path + '/' + name + '_' + str(index), data_map)

        del data_map

    return


def initialise_hebb(par):
    a_rnn = np.zeros((par['batch_size'], par['p_size'], par['p_size']))
    a_rnn_inv = np.zeros((par['batch_size'], par['p_size'], par['p_size']))
    return a_rnn, a_rnn_inv


def prepare_data_maps(data, prev_cell_maps, prev_acc_maps, positions, pars):
    gs, ps, ps_gen, x_s, position, acc_st = data
    gs_all, ps_all, ps_gen_all, xs_all = prev_cell_maps
    accs_x_to, accs_x_from = prev_acc_maps

    g1s = np.transpose(np.array(cp.deepcopy(gs)), [1, 2, 0])
    p1s = np.transpose(np.array(cp.deepcopy(ps)), [1, 2, 0])
    p1s_gen = np.transpose(np.array(cp.deepcopy(ps_gen)), [1, 2, 0])
    x_s1 = np.transpose(np.array(np.concatenate(np.transpose(cp.deepcopy(x_s), [1, 0, 2, 3]), axis=2)),
                        [1, 2, 0])  # clearly shouldn't need to use two transposes

    pos_to = position[:, 1:pars['seq_len'] + 1]
    pos_from = position[:, :pars['seq_len']]

    gs_all = cell_norm_online(g1s, pos_to, gs_all, pars)
    ps_all = cell_norm_online(p1s, pos_to, ps_all, pars)
    ps_gen_all = cell_norm_online(p1s_gen, pos_to, ps_gen_all, pars)
    xs_all = cell_norm_online(x_s1, pos_to, xs_all, pars)
    accs_x_to = accuracy_positions_online(acc_st, pos_to, accs_x_to, pars)
    accs_x_from = accuracy_positions_online(acc_st, pos_from, accs_x_from, pars)

    if positions is None:
        positions = [None] * pars['n_envs_save']
    positions = positions_online(position, positions, pars['n_envs_save'])

    cell_list = [gs_all, ps_all, ps_gen_all, xs_all]
    acc_list = [accs_x_to, accs_x_from]

    return acc_list, cell_list, positions


def prepare_cell_timeseries(data, prev_data, pars):
    gs, ps, poss = data
    gs_, ps_, pos_ = prev_data
    # convert to batch_size x cells x timesteps
    g1s = np.transpose(np.array(cp.deepcopy(gs)), [1, 2, 0])
    p1s = np.transpose(np.array(cp.deepcopy(ps)), [1, 2, 0])
    g1s = g1s[:pars['n_envs_save'], :, :]
    p1s = p1s[:pars['n_envs_save'], :, :]

    pos = poss[:, 1:]

    grids, places, positions = [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], [None] * pars['n_envs_save']

    for env in range(pars['n_envs_save']):
        try:
            grids[env] = np.concatenate((gs_[env], g1s[env]), axis=1)
            places[env] = np.concatenate((ps_[env], p1s[env]), axis=1)
            positions[env] = np.concatenate((pos_[env], pos[env]), axis=0)
        except:
            grids[env], places[env], positions[env] = cp.deepcopy(g1s[env]), cp.deepcopy(p1s[env]), \
                                                      cp.deepcopy(pos[env])

    return [grids, places, positions]


def combins(n, k, m):
    s = []
    for i in range(1, n + 1):
        c = scipy.special.comb(n - i, k)
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
    table_top = scipy.special.comb(n, k)
    for m in range(int(table_top)):
        # forward mapping
        c = combins(n, k, m)
        if map_max is None or m < map_max:
            table.append(c)
            rev_table[c] = m
        else:
            rev_table[c] = m % map_max
    return table, rev_table


def onehot2twohot(onehot, table, compress_size):
    seq_len = np.shape(onehot)[2]
    batch_size = np.shape(onehot)[0]
    twohot = np.zeros((batch_size, compress_size, seq_len))
    for i in range(np.shape(onehot)[2]):
        vals = np.argmax(onehot[:, :, i], 1)
        for b in range(np.shape(onehot)[0]):
            twohot[b, :, i] = table[vals[int(b)]]

    return twohot
