#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

from model_utils import DotDict
import data_utils
import importlib.util
import os
import scipy.stats as stats
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import parameters

fontsize = 25
linewidth = 4
labelsize = 20


def save_trained_outputs(date, run, index, use_old_scripts=True, base_path='../Summaries/', force_overwrite=False,
                         n_envs_save=6):
    """
    Load a trained model from a previous training run and save outputs
    """
    # Get directories for the requested run
    run_path, train_path, model_path, save_path, script_path, envs_path = \
        data_utils.set_directories(date, run, base_path=base_path)
    # Load model from file
    model, params = get_model(model_path, script_path, save_path, index, use_old_scripts=True)

    par_new = parameters.default_params()
    for key in par_new.keys():
        try:
            params[key]
        except (KeyError, AttributeError) as e:
            params[key] = par_new[key]
    try:
        params.env.four_actions
    except (KeyError, AttributeError) as e:
        params.env.four_actions = False

    # set n_envs_ssave to be high
    params.n_envs_save = n_envs_save
    # Make sure there is a trained model for the requested index (training iteration)
    if model is not None:
        # If the output directory already exists: only proceed if overwriting is desired
        dir_name = save_path + '/iter_' + str(index)
        if (os.path.exists(dir_name) and os.path.isdir(dir_name)) and not force_overwrite:
            if not os.listdir(dir_name):
                print('Running forward pass to collect data')
            else:
                print('Not running forward pass: ' + save_path + '/iter_' + str(index) + ' already exists')
                return model

        # Load data_utils from stored scripts of trained model (for compatibility) or current (for up-to-date)
        spec_data_utils = importlib.util.spec_from_file_location("data_utils", script_path + '/data_utils.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("data_utils", 'data_utils.py')
        stored_data_utils = importlib.util.module_from_spec(spec_data_utils)
        spec_data_utils.loader.exec_module(stored_data_utils)
        # Load model_utils from stored scripts of trained model
        spec_model_utils = importlib.util.spec_from_file_location("model_utils", script_path + '/model_utils.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("model_utils", 'model_utils.py')
        stored_model_utils = importlib.util.module_from_spec(spec_model_utils)
        spec_model_utils.loader.exec_module(stored_model_utils)
        # Create output folder
        if not os.path.exists(save_path + '/iter_' + str(index)):
            os.makedirs(save_path + '/iter_' + str(index))
        # Run forward pass and collect model outputs, then save model outputs to save forlder
        stored_data_utils.save_model_outputs(model, stored_model_utils, index, save_path + '/iter_' + str(index),
                                             params)
    return model


def get_model(model_path, script_path, save_path, index, use_old_scripts=True):
    """
    Load a trained model from a previous training run and save outputs
    """
    # Make sure there is a trained model for the requested index (training iteration)
    if os.path.isfile(model_path + '/tem_' + str(index) + '.index'):
        model_path = model_path + '/tem_' + str(index)
    elif os.path.isfile(model_path + '/tem_checkpoint_' + str(index) + '.index'):
        model_path = model_path + '/tem_checkpoint_' + str(index)
    else:
        print('Error: no trained model found for ' + model_path + '/tem_' + str(index) + '.index')
        print('Error: no trained model found for ' + model_path + '/tem_checkpoint_' + str(index) + '.index')
        # Return None to indicate error
        return None, None
    try:
        # Load model module from stored scripts of trained model
        spec_tem = importlib.util.spec_from_file_location("tem", script_path + '/tem_model.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("tem", 'tem_model.py')
        stored_tem = importlib.util.module_from_spec(spec_tem)
        spec_tem.loader.exec_module(stored_tem)
        # Load data_utils from stored scripts of trained model
        spec_data_utils = importlib.util.spec_from_file_location("data_utils", script_path + '/data_utils.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("data_utils", 'data_utils.py')
        stored_data_utils = importlib.util.module_from_spec(spec_data_utils)
        spec_data_utils.loader.exec_module(stored_data_utils)
        # Load model_utils from stored scripts of trained model
        spec_model_utils = importlib.util.spec_from_file_location("model_utils", script_path + '/model_utils.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("model_utils", 'model_utils.py')
        stored_model_utils = importlib.util.module_from_spec(spec_model_utils)
        spec_model_utils.loader.exec_module(stored_model_utils)
        # Load the parameters of the model
        params = stored_model_utils.DotDict(np.load(save_path + '/params.npy', allow_pickle=True).item())
        # Create a new tem model with the loaded parameters
        model = stored_tem.TEM(params)
        # Load the model weights after training
        model.load_weights(model_path)
        # Return loaded and trained model
        return model, params
    except ModuleNotFoundError as e:
        print('Model not found', e)
        return None, DotDict(np.load(save_path + '/params.npy', allow_pickle=True).item())


def link_inferences(data, env_dict, info, params):
    positions_link = data.pos_timeseries

    x = data.x_timeseries
    x_gt = data.x_gt_timeseries
    corrects_link = np.argmax(x, axis=1) == np.argmax(x_gt, axis=1)

    div = int(np.shape(corrects_link)[0] / np.shape(positions_link)[0])
    coo = np.split(corrects_link, div, axis=0)
    coos = np.concatenate(coo, 1)

    state_mat = [x.states_mat for x in env_dict.curric_env.envs]
    adjs = [x.adj for x in env_dict.curric_env.envs]
    wids = [x.width for x in env_dict.curric_env.envs]
    try:
        heights = [x.height for x in env_dict.curric_env.envs]
    except KeyError:
        heights = [None for _ in env_dict.curric_env.envs]

    n_states = np.asarray(
        [x.n_states if params.world_type != 'hexagonal' else x.graph_states for x in env_dict.curric_env.envs])

    n_available_states = np.zeros_like(wids)
    n_available_edges = np.zeros_like(wids)

    if len(adjs) == len(info.widths):
        for i, adj in enumerate(adjs):
            n_available_edges[i] = np.sum(np.sum(adj))
            n_available_states[i] = 0
            for j in range(len(adj)):
                if sum(adj[j, :]) > 0:
                    n_available_states[i] += 1
    else:
        for i, wid in enumerate(info.widths):
            adj_index = np.where(wid == info.widths[:len(adjs)])[0][:1][0]
            adj = adjs[adj_index]
            n_available_edges[i] = np.sum(np.sum(adj))
            n_available_states[i] = 0
            for j in range(len(adj)):
                if sum(adj[j, :]) > 0:
                    n_available_states[i] += 1

    info.n_states = n_states
    info.wids = wids
    info.heights = heights
    info.n_available_states = n_available_states
    info.n_available_edges = n_available_edges
    info.state_mat = state_mat
    info.state_guess = x_gt
    info.pos_dis = positions_link
    info.adjs = adjs

    return positions_link, coos, info


def analyse_link_inference(allowed_wid, positions_link, coos, info, params):
    p_cors = [[] for _ in range(params['batch_size'])]
    mean_cors = []
    nodes_visited_all, edges_visited_all = [], []
    n_visited_means, stay_still_corrects = [], []
    time_vis_anals, time_vis_inf_anals = [], []
    distance_anals, distance_random_anals = [], []
    transition_types = []

    for i in range(params['batch_size']):
        if info.wids[i] == allowed_wid:

            node_visited, edge_visited, nodes, edges, num_visited, edge_stay_still = \
                check_visited(positions_link[i].astype(int), info.n_states[i], info.stay_still_inf)
            nodes_visited_all.append(node_visited)
            edges_visited_all.append(edge_visited)

            time_vis_anal = time_since_visited(positions_link[i].astype(int), coos[i], info.a_s)
            time_vis_anals.append(time_vis_anal)

            time_vis_inf_anal = time_since_visited_inference(positions_link[i].astype(int), coos[i], node_visited,
                                                             edge_visited, info.a_s)
            time_vis_inf_anals.append(time_vis_inf_anal)

            n_visited_means.append(n_visited_correct(coos[i], num_visited, edge_visited))
            corrs, counters, corrs_counters = stay_still_correct(coos[i], num_visited, edge_stay_still)
            stay_still_corrects.append(corrs_counters)

            if info.state_guess is not None:
                distance_anal, distances_random_anal = check_wrong_distances(info.pos_dis[i].astype(int), coos[i],
                                                                             node_visited, info.state_guess[i],
                                                                             info.state_mat[i][:],
                                                                             info.adjs[i], params)
                distance_anals.append(distance_anal)
                distance_random_anals.append(distances_random_anal)

            transition_type = analyse_transition_type(positions_link[i].astype(int), coos[i], node_visited,
                                                      edge_visited, info.heights[i], info.wids[i], params)
            transition_types.append(transition_type)

            print('batch ' + str(i), sum(edge_visited), end=', ')
            mean_cor = None
            for j, (frac1, frac2) in enumerate(zip(info.fracs, info.fracs[1:])):
                p_cor, index, mean_cor = check_edge_inference(coos[i], node_visited, edge_visited, frac1, frac2,
                                                              info.n_available_states[i])
                p_cors[i].append(p_cor)
            mean_cors.append(mean_cor)

    p_cors = [ind for ind in p_cors if len(ind) > 0]

    results = DotDict({'p_cors': p_cors,
                       'nodes_visited_all': nodes_visited_all,
                       'edges_visited_all': edges_visited_all,
                       'time_vis_anals': time_vis_anals,
                       'stay_still_correct': stay_still_corrects,
                       'n_visited_means': n_visited_means,
                       'transition_types': transition_types,
                       'distance_anals': distance_anals,
                       'distance_random_anals': distance_random_anals,
                       'time_vis_inf_anals': time_vis_inf_anals,
                       'mean_cors': mean_cors
                       })

    return results


def check_visited(positions, states, stay_still_inf=True):
    """
    :param:
        positions: list of positions
    :return:
        node_visited: whether node visited starting at second position : 1 if never visited beofre
        edge_visited: whether edge to get to that node taken before : 1 if not taken before
        edges: all edges visited
        nodes: all nodes visited
    """
    # we care about 'second' node, and edge taken to reach it
    # returns lists

    max_len = len(str(max(positions))) + 2
    n_visited, num_visited, edge_stay_still = np.zeros(states), [], []
    node_visited, edge_visited, edges, nodes = [1], [], [], []  # start with one as we consider second not first
    str_pos = ''.join([',' + str(x + 0.0000001)[:max_len] + ',' for x in positions])

    for i, (first, second) in enumerate(zip(positions, positions[1:])):
        n_visited[first] += 1
        num_visited.append(n_visited[second])

        # node_visited - 1 not visited before, 0 if visited
        if second in positions[:i + 1]:
            node_visited.append(0)
        else:
            node_visited.append(1)
        nodes.append([second])

        first_str = str(first + 0.0000001)[:max_len]
        second_str = str(second + 0.0000001)[:max_len]

        if ',' + first_str + ',,' + second_str + ',' in str_pos[:(max_len + 2) * (i + 1)]:
            edge_visited.append(0)
        else:
            edge_visited.append(1)
        edges.append([first, second])

        if first_str == second_str and not stay_still_inf:
            print('checks')
            edge_visited[-1] = 0

        if first_str == second_str and ',' + first_str + ',,' + second_str + ',' not in str_pos[
                                                                                        :(max_len + 2) * (i + 1)]:
            edge_stay_still.append(1)  # new edge and stayed still
        else:
            edge_stay_still.append(0)

    return node_visited, edge_visited, nodes, edges, num_visited, edge_stay_still


def check_edge_inference(corrects, node_visited, edge_visited, frac1, frac2, states):
    """
    :param:
        corrects: Starts at correct for position 1
        node_visited: Starts at positions 1
        edge_visited: Starts at edge 1-2
        pars:
    :return: fraction of whether got prediction right on a visited node, but never visited edge

    """
    correct = corrects[1:]  # as we want correct for position 2 (edge 1-2)
    nodes = node_visited[1:]  # as we want correct for position 2 (edge 1-2)

    node_frac = np.cumsum(node_visited) / states

    counter = []
    corr_total = []

    for i, (corr, node, edge) in enumerate(zip(correct, nodes, edge_visited)):
        if node == 0 and edge == 1:  # node visited, but edge never taken before : inference of link
            counter.append(1)
            corr_total.append(corr)
        else:
            counter.append(0)
            corr_total.append(0)

    mean_total = sum(corr_total) / sum(counter)
    try:
        index1 = np.where(node_frac >= frac1)[0][0]
        try:
            index2 = np.where(node_frac >= frac2)[0][0]
        except IndexError:
            index2 = len(corr_total)

        return sum(corr_total[index1:index2]) / sum(counter[index1:index2]), [index1, index2], mean_total
    except (IndexError, ZeroDivisionError) as _:
        return float('NaN'), [float('NaN'), float('NaN')], mean_total


def n_visited_correct(corrects, num_visited, edge_visited):
    new_edges = np.where(np.asarray(edge_visited) == 1)[0]
    correct = corrects[1:]

    correct = correct[new_edges]
    num_visited = np.asarray(num_visited)[new_edges]

    means = []
    max_n = max(num_visited)
    for n in range(int(max_n)):
        means.append(np.mean(correct[np.where(np.asarray(num_visited) == n)]))
    return means


def check_wrong_distances(positions, corrects, node_visited, guesses, state_vec, adj, pars):
    rem_els = np.where(np.sum(adj, 0) == 0)[0]

    states = len(adj)
    width = int(np.sqrt(states))

    if len(state_vec.shape) > 1:
        state_vec = state_vec[:, 0]

    states_x, states_y = np.arange(states).astype(float) % width, np.floor(np.arange(states) / width)

    if pars['world_type'] in ['hex', 'hexagonal']:
        states_x += - 0.5 * np.mod(states_y, 2)
        states_y *= np.sqrt(3) / 2

    if rem_els.size > 0:
        states_x = np.delete(states_x, rem_els)
        states_y = np.delete(states_y, rem_els)
        state_vec = np.delete(state_vec, rem_els)

    states_xy = np.zeros((len(states_x), 2))
    states_xy[:, 0] = states_x
    states_xy[:, 1] = states_y

    # correct = corrects[1:]  # as we want correct for position 2 (edge 1-2)
    # nodes = node_visited[1:]  # as we want correct for position 2 (edge 1-2)

    distances = []
    distances_random = []

    for i, (corr, node, guess, pos) in enumerate(zip(corrects, node_visited, guesses, positions)):
        if node == 0 and corr == 0:  # node visited, wrong guess
            pos_x, pos_y = pos % width, int(pos / width)
            if pars['world_type'] in ['hex', 'hexagonal']:
                pos_x += - 0.5 * np.mod(pos_y, 2)
                pos_y *= np.sqrt(3) / 2

            distances.append(distance_to_guess(pos_x, pos_y, states_xy, state_vec, guess))

            distance_random = 0
            while distance_random == 0:  # random incorrect guess - this makes sure guess is incorrect!
                guess = np.random.randint(0, pars['s_size'])
                distance_random = distance_to_guess(pos_x, pos_y, states_xy, state_vec, guess)
            distances_random.append(distance_random)

    return distances, distances_random


def stay_still_correct(corrects, num_visited, edge_stay_still):
    # if stayed still, plot number of times node previously seen to accuracy of prediction
    correct = corrects[1:]
    corrs, counters, corrs_counters = [], [], []
    for num in range(1, 5):
        counter, corr = 0, 0
        for i, (x, y, z) in enumerate(zip(num_visited, edge_stay_still, correct)):
            if x == num and y == 1:
                counter += 1
                corr += z
        corrs.append(corr)
        counters.append(counter)
        corrs_counters.append(corr / (counter + 0.0001))
    return corrs, counters, corrs_counters


def time_since_vis(positions):
    lens = np.zeros_like(positions)
    for pos in np.unique(positions):
        where_pos = np.where(positions == pos)[0]  # find all locations in a state

        lens[where_pos[0]] = 1e6  # long time before first visit...
        if len(where_pos) > 1:
            differences = np.diff(where_pos)  # time between visits

            for posit, diff in zip(where_pos[1:], differences):
                lens[posit] = diff
    return lens


def time_since_visited_inference(positions, corrects, node_vis, edge_vis,
                                 a_s=(0, 10, 20, 40, 60, 100, 140, 200, 300, 400, 100000)):
    inf_pos = \
        np.where(np.logical_and((np.asarray(edge_vis).flatten() == 1), (np.asarray(node_vis[1:]).flatten() == 0)))[0]

    lens = time_since_vis(positions)

    lens = lens[inf_pos]
    corrects = corrects[inf_pos]

    res = []
    # corrects = coos[0]
    for diff in np.unique(lens):
        if diff < 10000:
            where_len = np.where(lens == diff)[0]
            res.append([diff, sum(corrects[where_len]), len(where_len)])

    res = np.asarray(res)

    re = []
    try:
        for a, b in zip(a_s, a_s[1:]):
            inde = np.where(np.logical_and(res[:, 0] >= a, res[:, 0] < b))[0]
            # print(inde)
            re.append([sum(res[inde, 1]), sum(res[inde, 2])])
    except IndexError:
        pass

    return re


def time_since_visited(positions, corrects, a_s):
    lens = time_since_vis(positions)

    res = []
    for diff in np.unique(lens):
        if diff < 10000:
            where_len = np.where(lens == diff)[0]
            res.append([diff, sum(corrects[where_len]), len(where_len)])

    res = np.asarray(res)

    res_fin = []
    for a, b in zip(a_s, a_s[1:]):
        inde = np.where(np.logical_and(res[:, 0] >= a, res[:, 0] < b))[0]
        res_fin.append([sum(res[inde, 1]), sum(res[inde, 2])])

    return res_fin


def analyse_transition_type(positions, corrects, node_visited, edge_visited, height, width, pars, env_class=None):
    correct = corrects[1:]  # as we want correct for position 2 (edge 1-2)
    nodes = node_visited[1:]  # as we want correct for position 2 (edge 1-2)

    transition_type_corrects = dict()
    transition_type_counts = dict()

    for i, (first, second, corr, node, edge) in enumerate(zip(positions, positions[1:], correct, nodes, edge_visited)):
        if node == 0 and edge == 1:  # node visited, but edge never taken before : inference of link
            rel_index, rel_type = env_class.relation(first, second)

            try:
                transition_type_counts[rel_type] += 1
            except KeyError:
                transition_type_counts[rel_type] = 1
                transition_type_corrects[rel_type] = 0

            if corr == 1:
                transition_type_corrects[rel_type] += 1

    # print(transition_type_counts, transition_type_corrects)
    transitions_summary = dict()
    for key, value in transition_type_counts.items():
        try:
            transitions_summary[key] = transition_type_corrects[key] / value
        except KeyError or ZeroDivisionError:
            pass

    return transitions_summary


def simulate_node_edge_agent(node_visited, edge_visited, acc=0.98, s_size=45):
    node_correct = []
    edge_correct = []
    for n, e in zip(node_visited, edge_visited):
        if n == 0:
            n_correct = stats.bernoulli.rvs(acc)
        else:
            n_correct = stats.bernoulli.rvs(1 / s_size)
        if e == 0:
            e_correct = stats.bernoulli.rvs(acc)
        else:
            e_correct = stats.bernoulli.rvs(1 / s_size)
        node_correct.append(n_correct)
        edge_correct.append(e_correct)

    return node_correct, edge_correct


def distance_to_guess(pos_x, pos_y, states_xy, state_vec, guess):
    # find closest state to pos which is same as guess
    guess_sense_states = np.where(state_vec == guess)[0]
    if guess_sense_states.size > 0:
        return min(np.sqrt(np.sum((states_xy[guess_sense_states] - [pos_x, pos_y]) ** 2, 1)))
    else:
        return 100


def sort_out_summin(n_visited_means):
    smallest = 10000
    for batch_means in n_visited_means:
        smallest = np.minimum(smallest, len(batch_means))
    print(smallest)
    for i, batch_means in enumerate(n_visited_means):
        n_visited_means[i] = batch_means[:smallest]

    return n_visited_means, smallest


def smooth(a, wsz):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    out0 = np.convolve(a, np.ones(wsz, dtype=int), 'valid') / wsz
    r = np.arange(1, wsz - 1, 2)
    start = np.cumsum(a[:wsz - 1])[::2] / r
    stop = (np.cumsum(a[:-wsz:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def plot_link_inference(results, info):
    # plot as a function of how many times seen target node to test hebb learning rate
    # change environments more often to learn better prior etc, as soon as seen all nodes once

    f = plt.figure()

    for res, wid in zip(results, info.allowed_widths):
        pc = res.p_cors
        mean_pc = np.nanmean(pc, 0)
        std_pc = np.nanstd(pc, 0)
        plt.fill_between(info.fracs[:-1], mean_pc + std_pc, mean_pc - std_pc, alpha=0.2)
        plt.plot(info.fracs[:-1], mean_pc, linewidth=linewidth, label=str(wid))

    plt.plot((0, 1), (1 / info.s_size, 1 / info.s_size), 'k--', linewidth=linewidth)

    plt.ylim(0, 1.1)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tick_params(axis='both', which='minor', labelsize=labelsize)
    plt.xlabel('Proportion of nodes visited', fontsize=fontsize)
    plt.ylabel('Correct inference of link', fontsize=fontsize)
    plt.show()

    f.savefig('./figures/' + "link_inference.pdf", bbox_inches='tight')

    return


def plot_acc_vs_sum_nodes_edges(results, coos, info, cutoff=None):
    if len(results) > 1:
        f = plt.figure(figsize=(18, 5))
    else:
        f = plt.figure(figsize=(5, 5))
    filt = None
    for i, (res, wid) in enumerate(zip(results, info.allowed_widths)):
        node_visited_all = res.nodes_visited_all
        edge_visited_all = res.edges_visited_all

        coos_ = coos[info.wids == wid]
        coo_mean = np.mean(coos_, 0)

        # SHOULD PLOT TEM VS NODE/EDGE AGENT!

        filt = smooth(coo_mean, info.filt_size)
        plt.subplot(1, len(np.unique(info.allowed_widths)), i + 1)
        plt.plot(filt[:cutoff], 'k', alpha=0.4, label='TEM', linewidth=linewidth)

        nodes_visited = np.cumsum(
            np.mean(np.transpose([x / info.n_available_states[info.wids == wid] for x in np.transpose(
                node_visited_all)]), 0))
        edges_visited = np.cumsum(
            np.mean(np.transpose([x / info.n_available_edges[info.wids == wid] for x in np.transpose(
                edge_visited_all)]), 0))

        plt.plot(smooth(nodes_visited[:cutoff], info.filt_size), 'r--', label='Proportion of nodes visited',
                 linewidth=linewidth)
        plt.plot(smooth(edges_visited[:cutoff], info.filt_size), 'b--', label='Proportion of edges visited',
                 linewidth=linewidth)

        plt.xlabel('# steps taken', fontsize=fontsize)
        plt.ylabel('Prediction accuracy', fontsize=fontsize)

        plt.ylim([0, 1.1])
        plt.legend(prop={'size': 15})
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        plt.tick_params(axis='both', which='minor', labelsize=labelsize)

    plt.plot((0, len(filt)), (1 / info.s_size, 1 / info.s_size), 'k--', linewidth=linewidth)
    plt.plot((0, len(filt)), (1, 1), 'k--', linewidth=linewidth)

    plt.show()

    f.savefig('./figures/' + "acc_vs_sum_nodes.pdf", bbox_inches='tight')

    return


def plot_sim_corrects(results, info):
    if len(results) > 1:
        f = plt.figure(figsize=(18, 5))
    else:
        f = plt.figure(figsize=(5, 5))
    n_coo_mean = None
    for i, (res, wid) in enumerate(zip(results, info.allowed_widths)):
        node_visited_all = res.nodes_visited_all
        edge_visited_all = res.edges_visited_all

        node_correct_, edge_correct_ = [], []
        for node_visited, edge_visited in zip(node_visited_all, edge_visited_all):
            node_correct, edge_correct = simulate_node_edge_agent(node_visited, edge_visited, acc=info.acc_simu,
                                                                  s_size=info.s_size)
            node_correct_.append(node_correct)
            edge_correct_.append(edge_correct)

        n_coo_mean = np.mean(node_correct_, 0)
        e_coo_mean = np.mean(edge_correct_, 0)

        plt.subplot(1, len(np.unique(info.allowed_widths)), i + 1)
        plt.plot(smooth(n_coo_mean, info.filt_size), 'r', alpha=0.4, label='Node agent', linewidth=linewidth)
        plt.plot(smooth(e_coo_mean, info.filt_size), 'b', alpha=0.4, label='Edge agent', linewidth=linewidth)

        nodes_visited = np.cumsum(
            np.mean(np.transpose([x / info.n_available_states[info.wids == wid] for x in np.transpose(
                node_visited_all)]), 0))
        edges_visited = np.cumsum(
            np.mean(np.transpose([x / info.n_available_edges[info.wids == wid] for x in np.transpose(
                edge_visited_all)]), 0))

        plt.plot(smooth(nodes_visited, info.filt_size), 'r--', label='Proportion of nodes visited', linewidth=linewidth)
        plt.plot(smooth(edges_visited, info.filt_size), 'b--', label='Proportion of edges visited', linewidth=linewidth)

        plt.xlabel('# steps taken', fontsize=fontsize)
        plt.ylabel('Prediction accuracy', fontsize=fontsize)

        plt.ylim([0, 1.1])
        plt.legend(prop={'size': 15})
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        plt.tick_params(axis='both', which='minor', labelsize=labelsize)

    plt.plot((0, len(n_coo_mean)), (1 / info.s_size, 1 / info.s_size), 'k--', linewidth=linewidth)
    plt.plot((0, len(n_coo_mean)), (1, 1), 'k--', linewidth=linewidth)

    plt.show()

    f.savefig('./figures/' + "acc_vs_sum_nodes_simulated.pdf", bbox_inches='tight')

    return


def plot_acc_vs_steps_since_visited(results, info, first_time=False):
    f = plt.figure()

    edges = [(x + y) / 2 for x, y in zip(info.a_s, info.a_s[1:])]

    for i, (res, wid) in enumerate(zip(results, info.allowed_widths)):
        s_s_c = np.asarray(res.time_vis_anals) if not first_time else np.asarray(res.time_vis_inf_anals)
        s_s_c = s_s_c[:, :, 0] / s_s_c[:, :, 1]
        mean_ssc = np.nanmean(s_s_c, 0)
        std_ssc = np.nanstd(s_s_c, 0)
        plt.plot(edges, mean_ssc, label=str(wid), linewidth=linewidth)

        plt.fill_between(edges, mean_ssc + std_ssc, mean_ssc - std_ssc, alpha=0.2)
    plt.xlabel('# steps since visited', fontsize=fontsize)
    plt.ylabel('Prediction accuracy', fontsize=fontsize)
    plt.plot((edges[0], edges[-1]), (1 / info.s_size, 1 / info.s_size), 'k--', linewidth=linewidth)
    plt.ylim(0, 1.1)
    plt.xticks(edges[0:1] + edges[5:-1])

    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tick_params(axis='both', which='minor', labelsize=labelsize)
    plt.legend(fontsize=fontsize)
    plt.show()

    f.savefig('./figures/' + "beyond_bptt.pdf", bbox_inches='tight')

    return


def plot_seq_position_accuracy(coos, info):
    aa = np.mean(np.split(coos, axis=1, indices_or_sections=int(np.shape(coos)[1] / info.seq_len)), 0)
    for i, wid in enumerate(info.allowed_widths):
        seq = np.mean(aa[info.wids == wid, :], 0)
        plt.plot(seq, label=str(wid))
    plt.plot(np.mean(aa, 0), label='all')
    plt.legend()
    plt.show()

    return


def plot_acc_curves_visited(results, coos, info):
    min_len = []
    for i, (res, wid) in enumerate(zip(results, info.allowed_widths)):
        node_vis = res.nodes_visited_all
        coo_wid = coos[info.wids == wid]

        visited = list(np.where(np.asarray(node_vis) == 0))
        # in case not every state has been visited
        a = []
        last_index = []
        for j in np.unique(visited[0]):
            a.append(np.sum(visited[0] == j))
            last_index.append(np.sum(visited[0] <= j) - 1)
        diffs = np.array(a) - np.min(a)
        # remove diff number of elements from each to make the same length
        for j, diff in reversed(list(enumerate(diffs))):
            if diff > 0:
                for d in range(diff):
                    visited[0] = np.delete(visited[0], last_index[j] - d)
                    visited[1] = np.delete(visited[1], last_index[j] - d)

        visited = tuple(visited)

        meaned = np.mean(np.reshape(coo_wid[visited], [np.shape(coo_wid)[0], -1]),
                         0)

        filt = smooth(meaned, info.filt_size)

        plt.plot(filt, label=str(wid))

        min_len.append(len(filt))
    min_len = np.min(min_len) - 10
    plt.legend()
    plt.xlabel('# steps', fontsize=fontsize)
    plt.ylabel('Prediction accuracy', fontsize=fontsize)
    plt.plot((0, min_len), (1 / info.s_size, 1 / info.s_size), 'k--', linewidth=linewidth)
    plt.ylim(0, 1.1)
    plt.xlim(0, min_len)
    plt.show()

    return


def plot_distance_analysis(results, info):
    f = plt.figure(figsize=(12, 10))
    if not info.rem_100:
        info.bins.append(101)
    for i, (res, wid) in enumerate(zip(results, info.allowed_widths)):
        dists = np.around(np.hstack(res.distance_anals), decimals=2)
        dists_random = np.around(np.hstack(res.distance_random_anals), decimals=2)

        len_dists = len(dists)

        if info.rem_100:
            dists = np.delete(dists, np.where(dists == 100)[0])
            dists_random = np.delete(dists_random, np.where(dists_random == 100)[0])

        print('width', wid)
        print('proportion of incorrect guesses in environment', len(dists) / len_dists, len(dists_random) / len_dists)
        print('mean distance from correct', np.mean(dists), np.mean(dists_random))

        plt.subplot(2, len(np.unique(info.allowed_widths)), i + 1)
        _, _, _ = plt.hist(dists, bins=info.bins, facecolor='blue', alpha=0.4, label='Data')
        _, _, _ = plt.hist(dists_random, bins=info.bins, facecolor='green', alpha=0.4, label='Random')
        plt.legend(fontsize=fontsize)

        plt.subplot(2, len(np.unique(info.allowed_widths)), i + 4)
        sns.distplot(dists, bins=info.bins, kde=True, label='Data')
        sns.distplot(dists_random, bins=info.bins, kde=True, label='Random')
        plt.legend(fontsize=fontsize)

    plt.show()

    f.savefig('./figures/' + "plot_distance_analysis.pdf", bbox_inches='tight')

    return


def plot_relation_type_inference_acc(results, info):
    # inference analysis for different directions taken

    plt.figure(figsize=(18, 5))
    for i, (res, wid) in enumerate(zip(results, info.allowed_widths)):
        accs = res.transition_types

        plt.subplot(1, len(np.unique(info.allowed_widths)), i + 1)

        x = []
        mean = []
        err = []
        for key, value in accs[0].items():
            vals = []
            for acc in accs:
                try:
                    vals.append(acc[key])
                except KeyError:
                    pass
            mean.append(np.mean(vals))
            err.append(np.std(vals))
            x.append(key)

        indices = list(np.argsort(x))
        x = [x[j] for j in indices]
        mean = [mean[j] for j in indices]
        err = [err[j] for j in indices]

        x_axis = np.arange(len(x))
        plt.errorbar(x_axis, mean, yerr=err, fmt='o')
        plt.plot((np.minimum(np.min(x_axis), 0), np.max(x_axis)), (1 / info.s_size, 1 / info.s_size), 'k--',
                 linewidth=linewidth)
        plt.ylim([0, 1.1])

        plt.xticks(np.arange(len(x)), x, rotation='vertical')

    plt.show()

    return


def plot_stay_still_correct(results, info):
    f = plt.figure()

    for i, (res, wid) in enumerate(zip(results, info.allowed_widths)):
        s_s_c = res.stay_still_correct

        mean_ssc = np.nanmean(s_s_c, 0)
        std_ssc = np.nanstd(s_s_c, 0)
        plt.plot(np.arange(1, 5), mean_ssc, label=str(wid), linewidth=linewidth)

        plt.fill_between(np.arange(1, 5), mean_ssc + std_ssc, mean_ssc - std_ssc, alpha=0.2)
    plt.xlabel('# times node re-visited', fontsize=fontsize)
    plt.ylabel('Prediction accuracy', fontsize=fontsize)
    plt.plot((1, 4), (1 / info.s_size, 1 / info.s_size), 'k--', linewidth=linewidth)
    plt.ylim(0, 1.1)
    plt.xticks(np.arange(1, 5))
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tick_params(axis='both', which='minor', labelsize=labelsize)
    plt.legend(fontsize=fontsize)
    plt.show()

    f.savefig('./figures/' + "stay_still_corrects.pdf", bbox_inches='tight')

    return


def plot_inference_vs_num_node_visited(results, info):
    f = plt.figure()
    smallest_min = 10000
    for res, wid in zip(results, info.allowed_widths):
        pc = res.n_visited_means
        pc, smallest = sort_out_summin(pc)
        smallest_min = np.minimum(smallest, smallest_min)

        mean_pc = np.nanmean(pc, 0)
        std_pc = np.nanstd(pc, 0)
        plt.plot(mean_pc)
        plt.fill_between(np.arange(smallest), mean_pc + std_pc, mean_pc - std_pc,
                         alpha=0.2, label=str(wid))

    plt.plot((0, smallest_min - 1), (1 / info.s_size, 1 / info.s_size), 'k--', linewidth=linewidth)
    plt.xlim(0, smallest_min - 1)
    plt.ylim(0, 1.1)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tick_params(axis='both', which='minor', labelsize=labelsize)
    plt.xlabel('# times node visited', fontsize=fontsize)
    plt.ylabel('Correct inference of link', fontsize=fontsize)
    plt.legend()
    plt.show()

    f.savefig('./figures/' + "num_node_visited.pdf", bbox_inches='tight')

    return


def plot_acc_curves(results, coos, info):
    filt = None
    for res, wid in zip(results, info.allowed_widths):
        coo_mean = np.mean(coos[info.wids == wid], 0)

        filt = smooth(coo_mean, info.filt_size)
        plt.plot(filt, label=str(wid))

    plt.plot((0, len(filt)), (1 / info.s_size, 1 / info.s_size), 'k--', linewidth=linewidth)

    plt.legend()
    plt.show()

    return
