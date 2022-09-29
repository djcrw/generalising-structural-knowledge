#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

from parameters import *
from helper_functions import *
import copy as cp
import tensorflow as tf
import numpy as np
import importlib
import tem_model as tem

importlib.reload(tem)

gen_path, train_path, model_path, save_path, script_path = make_directories()
logger = make_logger(gen_path)

""" MODEL """
pars = default_params()
save_params(pars, save_path, script_path)
pars_orig = pars.copy()

tf.reset_default_graph()
print('Initialising graph')
with tf.name_scope('Inputs'):
    it_num = tf.placeholder(tf.float32, shape=(), name='it_num')
    seq_ind = tf.placeholder(tf.float32, shape=(), name='seq_ind')
    rnn = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['p_size'], pars['p_size']), name='rnn')
    rnn_inv = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['p_size'], pars['p_size']), name='rnn')
    x1_two_hot = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['s_size_comp'], pars['seq_len']),
                                name='x1_two_hot')
    x1 = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['s_size'], pars['seq_len']), name='x')
    g_ = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['g_size']), name='g_')
    x_ = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['s_size_comp'] * pars['n_freq']), name='x_')
    sh = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['s_size']), name='shiny')
    # need to feed in lists etc
    d0 = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['n_actions'], pars['seq_len']), name='d')
    s_visi = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['seq_len']), name='s_visited')
    no_d = tf.placeholder(tf.float32, shape=(pars['batch_size']), name='no_direc_batch')
    x_two_hot = tf.unstack(x1_two_hot, axis=2)
    x = tf.unstack(x1, axis=2)
    d = tf.unstack(d0, axis=2)
    s_vis = tf.unstack(s_visi, axis=1)

model = tem.TEM(x, x_, x_two_hot, g_, d, rnn, rnn_inv, it_num, seq_ind, s_vis, sh, no_d, pars)

fetches_all, fetches_summary, fetches_all_, fetches_summary_ = [], [], [], []
fetches_all.extend([model.g, model.p, model.p_g, model.x_gt, model.x_, model.A, model.A_inv, model.accuracy_gt,
                    model.train_op_all])
fetches_all_.extend([model.g, model.p, model.p_g, model.x_gt, model.x_, model.A, model.A_inv, model.accuracy_gt,
                     model.temp])
fetches_summary.extend([model.lx_p, model.lx_g, model.lx_gt, model.lp, model.lg, model.g, model.p, model.p_g,
                        model.x_gt, model.x_, model.A, model.A_inv, model.accuracy_p, model.accuracy_g,
                        model.accuracy_gt, model.merged, model.train_op_all])
fetches_summary_.extend([model.lx_p, model.lx_g, model.lx_gt, model.lp, model.lg, model.g, model.p, model.p_g,
                         model.x_gt, model.x_, model.A, model.A_inv, model.accuracy_p, model.accuracy_g,
                         model.accuracy_gt, model.merged, model.temp])

print('Graph initialised')

# CREATE SESSION
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=1)  # saves variables learned during training
train_writer = tf.summary.FileWriter(train_path, sess.graph)
tf.global_variables_initializer().run()
tf.get_default_graph().finalize()

""" RUN MODEL """
# INITIALISE VARIABLES
lx_ps, lx_gs, lx_gts, lps, lgs = [], [], [], [], []
accs_p, accs_g, accs_gt = [], [], []
check_link_inference = False
acc_p, acc_g, acc_gt, seq_index, rn = 0, 0, 0, 0, 0
correct_link, positions_link, positions, visited, state_guess = None, None, None, None, None
position_all, direc_all = None, None
gs_all, ps_all, ps_gen_all, xs_all, gs_timeseries, ps_timeseries, pos_timeseries = \
    [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], \
    [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], \
    [None] * pars['n_envs_save']
cell_timeseries, prev_cell_timeseries = None, None
accs_x_to, accs_x_from = [None] * pars['n_envs_save'], [None] * pars['n_envs_save']
save_needed, save_ticker, summary_needed, summary_ticker, save_model = False, False, False, False, False
table, _ = combins_table(pars['s_size_comp'], 2)
n_restart = pars['restart_max'] + pars['curriculum_steps']
T, F, L, P = None, None, None, None
adjs, trans, states_mat, shiny_s, shiny_states = None, None, None, None, None
a_rnn, a_rnn_inv = None, None
gs, ps, x_s, x_data, start_state, prev_direc = None, None, None, None, None, None
n_walk = None
index = 0
# width of env for particular batch is pars['widths'][pars['diff_env_batches_envs'][env]]

print('Training Started') if pars['training'] else print('Debugging Started')
for i in range(pars['train_iters']):

    # INITIALISE ENVIRONMENT AND INPUT VARIABLES
    msg = 'New Environment ' + str(i) + ' ' + str(index) + ' ' + str(index * pars['seq_len'])
    logger.info(msg)

    # curriculum of behaviour types
    pars, shiny_s, rn, n_restart, no_direc_batch = curriculum(pars_orig, pars, n_restart)

    if save_ticker:
        save_needed = True
        gs_all, ps_all, ps_gen_all, xs_all, gs_timeseries, ps_timeseries, pos_timeseries = \
            [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], \
            [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], [None] * pars['n_envs_save'], \
            [None] * pars['n_envs_save']
        accs_x_to, accs_x_from = [None] * pars['n_envs_save'], [None] * pars['n_envs_save']
        n_walk = pars['n_save_data']
    elif i % 10 in [5]:  # check for link inference every 10 environments
        check_link_inference, positions_link, correct_link, state_guess = True, [None] * pars['batch_size'], \
                                                                          [None] * pars['batch_size'], [None] * pars[
                                                                              'batch_size']
        n_walk = pars['link_inf_walk']
    else:
        n_walk = int(n_restart) + rn

    # make environemnts
    adjs, trans, states_mat, shiny_states = make_environments(pars)

    # initialise Hebbian matrices
    a_rnn, a_rnn_inv = initialise_hebb(pars)

    # initialise all other variables
    gs, x_s, x_data, start_state, prev_direc, visited = initialise_variables(pars, adjs)

    # Collect full sequence of data
    position_all, direc_all = get_walking_data(start_state, adjs, trans, prev_direc, shiny_states,
                                               n_walk * pars['n_walk'], pars)

    # run model
    for seq_index in range(n_walk):
        # summary at end of each walk
        summary_needed = True if seq_index == n_walk - 1 and summary_ticker else False

        # COLLECT DATA
        i1, i2 = seq_index * pars['n_walk'], (seq_index + 1) * pars['n_walk'] + 1
        walking_data = [position_all[:, i1:i2], direc_all[:, :, i1:i2 - 1]]
        new_data, old_data, model_vars = get_next_batch(walking_data, x_data, states_mat, index, visited, pars)
        xs, ds, position, visited, s_visited = new_data
        x_data, start_state = old_data
        T, F, L, P = model_vars
        xs_two_hot = onehot2twohot(xs, table, pars['s_size_comp'])

        feed_dict = {x1: xs, x_: x_s, x1_two_hot: xs_two_hot, g_: gs, d0: ds, rnn: a_rnn, rnn_inv: a_rnn_inv,
                     it_num: index, seq_ind: seq_index, s_visi: s_visited, sh: shiny_s, no_d: no_direc_batch}

        # run model +/- summaries
        if summary_needed:
            # don't train when getting representations or link inference
            fetch = fetches_summary_ if check_link_inference or save_needed else fetches_summary
            results = sess.run(fetch, feed_dict)
            lx_p, lx_g, lx_gt, lp, lg, gs, ps, ps_gen, x_gt, x_s, a_rnn, a_rnn_inv, acc_p, acc_g, acc_gt, summary, \
                _ = results

            train_writer.add_summary(summary, index * pars['seq_len'])
            accs_p.append(acc_p)
            accs_g.append(acc_g)
            accs_gt.append(acc_gt)
            lx_ps.append(lx_p)
            lx_gs.append(lx_g)
            lx_gts.append(lx_gt)
            lps.append(lp)
            lgs.append(lg)
        else:
            fetch = fetches_all_ if check_link_inference or save_needed else fetches_all
            results = sess.run(fetch, feed_dict)
            gs, ps, ps_gen, x_gt, x_s, a_rnn, a_rnn_inv, acc_gt, _ = results

        # Check for nans etc.
        for ar, array in enumerate([gs, ps, ps_gen, x_gt, x_s]):
            if np.isnan(array).any():
                raise ValueError('Nan in array ' + str(ar))

        # checking link inference
        if check_link_inference:
            # store positions and store g_t accuracy at each position
            positions_link = positions_online(position, positions_link, pars['batch_size'])
            _, correct_link = accuracy_online(correct_link, acc_sense, xs, x_gt, pars['seq_len'])
            state_guess = sense_online(x_gt, state_guess, pars['seq_len'])

        # preparing representations for saving
        if save_needed:
            acc_st, _ = accuracy_online(None, acc_sense, xs, x_gt, pars['seq_len'])
            save_data = [gs, ps, ps_gen, x_s, position, acc_st]
            prev_cell_maps = [gs_all, ps_all, ps_gen_all, xs_all]
            prev_acc_maps = [accs_x_to, accs_x_from]

            acc_list, cell_list, positions = prepare_data_maps(save_data, prev_cell_maps, prev_acc_maps, positions,
                                                               pars)
            gs_all, ps_all, ps_gen_all, xs_all = cell_list
            accs_x_to, accs_x_from = acc_list

            prev_cell_timeseries = [gs_timeseries, ps_timeseries, pos_timeseries]
            save_data_timeseries = [gs, ps, position]
            cell_timeseries = prepare_cell_timeseries(save_data_timeseries, prev_cell_timeseries, pars)
            gs_timeseries, ps_timeseries, pos_timeseries = cell_timeseries

        if index % pars['save_interval'] == 0 and index > 0:
            save_ticker = True
        if index % pars['save_model'] == 0:
            save_model = True
        if index % pars['sum_int'] == 0:
            summary_ticker = True
        index += 1

        # feeding in correct initial states
        gs, x_s = cp.deepcopy(gs[-1]), cp.deepcopy(np.concatenate(x_s[-1], 1))

    # save representations
    if save_needed:
        states = [pars['n_states_world'][pars['diff_env_batches_envs'][env]] for env in range(pars['n_envs_save'])]

        data_list = [gs_all, ps_all, ps_gen_all, xs_all, accs_x_to, accs_x_from]
        names = ['g_all', 'p_all', 'p_gen_all', 'x_all', 'acc_s_t_to', 'acc_s_t_from']
        save_data_maps(positions, data_list, save_path, pars['n_envs_save'], index, states, names)

        g2g = [v for v in tf.trainable_variables() if "mu_g2g" in v.name][0]
        np.save(save_path + '/A_RNN_' + str(index), a_rnn[:pars['n_envs_save']])
        np.save(save_path + '/A_RNN_inv_' + str(index), a_rnn_inv[:pars['n_envs_save']])
        np.save(save_path + '/g2g_' + str(index), g2g.eval())
        np.save(save_path + '/shiny_states_' + str(index), shiny_states)
        np.save(save_path + '/widths_' + str(index), pars['widths'])
        np.save(save_path + '/adj_' + str(index), adjs[:pars['n_envs_save']])
        np.save(save_path + '/gs_timeseries_' + str(index), gs_timeseries)
        if pars['world_type'] == 'loop_laps':
            np.save(save_path + '/ps_timeseries_' + str(index), ps_timeseries)
        np.save(save_path + '/pos_timeseries_' + str(index), pos_timeseries)
        save_needed, save_ticker = False, False
        del data_list, gs_all, ps_all, ps_gen_all, xs_all, accs_x_to, accs_x_from, gs_timeseries, pos_timeseries, \
            ps_timeseries, cell_timeseries, prev_cell_timeseries
        positions = None

    # save link inferences
    if check_link_inference:
        np.save(save_path + '/positions_link' + str(index), positions_link)
        np.save(save_path + '/correct_link' + str(index), correct_link)
        np.save(save_path + '/state_mat_link' + str(index), states_mat)
        np.save(save_path + '/state_guess_link' + str(index), state_guess)
        np.save(save_path + '/adj_link' + str(index), adjs)
        check_link_inference = False
        del positions_link, correct_link, state_guess

    # save model
    if save_model:
        saver.save(sess, model_path + '/TTJmodel' + str(index) + '.ckpt')
        save_model = False

    # log summary stats
    if summary_needed:  # only summary after been in environment for a while
        msg = "MaxHebb={:.5f}, T={:.2f}, F={:.2f}, L={:.2f}, i={:.2f}, index={:.2f}".format(
            np.max(np.max(np.abs(a_rnn))), T, F, L, index, index * pars['seq_len'])
        logger.info(msg)
        msg = ("it={:.5f}, lxP={:.2f}, lxG={:.2f}, lxGt={:.2f}, lp={:.2f}, lg={:.2f}, aP={:.2f}, aGinf={:.2f}, "
               + "aGt={:.2f}").format(index, lx_ps[-1], lx_gs[-1], lx_gts[-1], lps[-1], lgs[-1],
                                      acc_p, acc_g, acc_gt)
        logger.info(msg)
        summary_needed, summary_ticker = False, False

train_writer.close()
sess.close()
print('Finished training')

# SAVE DATA
markers = [lx_ps, lx_gs, lx_gts, lps, lgs, accs_p, accs_g, accs_gt]
np.save(save_path + '/markers', markers)
