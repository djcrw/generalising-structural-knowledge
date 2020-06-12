#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

from plotting_functions import *
from environment_functions import *
import scipy.stats as stats

fontsize = 25
linewidth = 4
labelsize = 20


def link_inferences(save_path, list_of_files, widths, batch_id, params, index=-1):
    list_of_files_2 = [x for x in list_of_files if 'positions_link' in x]

    list_of_files_3 = [int(x.split('.')[0].split('link')[-1]) for x in list_of_files_2]
    list_of_files_3.sort()
    index = str(list_of_files_3[index])

    print(index, len(list_of_files_3))

    positions_link = np.load(save_path + '/positions_link' + index + '.npy')
    corrects_link = np.load(save_path + '/correct_link' + index + '.npy')
    state_mat = np.load(save_path + '/state_mat_link' + index + '.npy')
    state_guess = np.load(save_path + '/state_guess_link' + index + '.npy')
    pos_dis = np.load(save_path + '/positions_link' + index + '.npy')
    adjs = np.load(save_path + '/adj_link' + index + '.npy')

    div = int(np.shape(corrects_link)[0] / np.shape(positions_link)[0])
    coo = np.split(corrects_link, div, axis=0)
    coos = np.concatenate(coo, 1)

    wids = np.asarray(widths)[batch_id]

    if params['world_type'] == 'family_tree':
        n_states = 2 ** (wids + 1) - 1
    elif params['world_type'] == 'line_ti':
        n_states = wids
    elif params['world_type'] == 'loop_laps':
        n_states = params['n_laps'] * (2 * wids + 2 * (wids - 1))
    elif params['world_type'] == 'rectangle':
        n_states = wids * np.asarray(params['heights'])[batch_id]
    else:
        n_states = wids**2

    n_available_states = np.zeros_like(wids)
    n_available_edges = np.zeros_like(wids)
    for i, adj in enumerate(adjs):
        n_available_edges[i] = np.sum(np.sum(adj))
        n_available_states[i] = 0
        for j in range(len(adj)):
            if sum(adj[j, :]) > 0:
                n_available_states[i] += 1

    env_info = [n_states, wids, n_available_states, n_available_edges]

    return positions_link, coos, env_info, [state_mat, state_guess, pos_dis, adjs]


def analyse_link_inference(allowed_wid, fracs, a_s, positions_link, coos, env_info, params):
    n_states, wids, n_available_states, n_available_edges = env_info

    p_cors = [[] for _ in range(params['batch_size'])]
    indexs = [[] for _ in range(params['batch_size'])]
    nodes_visited_all, edges_visited_all = [], []
    time_vis_anals = []

    for i in range(params['batch_size']):
        if wids[i] == allowed_wid:
            params['states'] = n_states[i]
            params['n_available_states'] = n_available_states[i]
            params['edges'] = n_available_edges[i]

            node_visited, edge_visited, nodes, edges, num_visited, edge_stay_still = \
                check_visited(positions_link[i].astype(int), params)

            nodes_visited_all.append(node_visited)
            edges_visited_all.append(edge_visited)

            time_vis_anal = time_since_visited(positions_link[i].astype(int), coos[i], a_s)
            time_vis_anals.append(time_vis_anal)

            print('batch ' + str(i), sum(edge_visited), end=', ')
            for j, (frac1, frac2) in enumerate(zip(fracs, fracs[1:])):
                    p_cor, index = check_edge_inference(coos[i], node_visited, edge_visited, frac1, frac2, params)
                    p_cors[i].append(p_cor)
                    indexs[i].append(index)

    return p_cors, nodes_visited_all, edges_visited_all, time_vis_anals


def check_visited(positions, pars):
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
    n_visited, num_visited, edge_stay_still = np.zeros(pars['states']), [], []
    node_visited, edge_visited, edges, nodes = [1], [], [], []  # start with one as we consider second not first
    str_pos = ''.join([',' + str(x + 0.0000001)[:max_len] + ',' for x in positions])

    for i, (first, second) in enumerate(zip(positions, positions[1:])):
        n_visited[first] += 1
        num_visited.append(n_visited[second])

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

        if first_str == second_str and ',' + first_str + ',,' + second_str + ',' not in str_pos[
                                                                                        :(max_len + 2) * (i + 1)]:
            edge_stay_still.append(1)  # new edge and stayed still
        else:
            edge_stay_still.append(0)

    return node_visited, edge_visited, nodes, edges, num_visited, edge_stay_still


def check_edge_inference(corrects, node_visited, edge_visited, frac1, frac2, pars):
    """
    :param:
        corrects: Starts at correct for position 1
        node_visited: Starts at positions 1
        edge_visited: Starts at edge 1-2
        pars:
    :return: fraction of whether got prediction right on a visited node, but never visited edge

    """
    states = pars['n_available_states']  # do this properly
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
    try:
        index1 = np.where(node_frac >= frac1)[0][0]
        try:
            index2 = np.where(node_frac >= frac2)[0][0]
        except:
            index2 = len(corr_total)

        return sum(corr_total[index1:index2]) / sum(counter[index1:index2]), [index1, index2]
    except:
        return float('NaN'), [float('NaN'), float('NaN')]


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
    out0 = np.convolve(a, np.ones(wsz, dtype=int), 'valid')/wsz
    r = np.arange(1, wsz-1, 2)
    start = np.cumsum(a[:wsz-1])[::2]/r
    stop = (np.cumsum(a[:-wsz:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


def plot_link_inference(results, allowed_widths, fracs, s_size):
    # plot as a function of how many times seen target node to test hebb learning rate
    # change environments more often to learn better prior etc, as soon as seen all nodes once

    f = plt.figure()

    for res, wid in zip(results, allowed_widths):
        pc = res[0]
        mean_pc = np.nanmean(pc, 0)
        std_pc = np.nanstd(pc, 0)
        plt.fill_between(fracs[:-1], mean_pc + std_pc, mean_pc - std_pc, alpha=0.2)
        plt.plot(fracs[:-1], mean_pc, linewidth=linewidth, label=str(wid))

    plt.plot((0, 1), (1/s_size, 1/s_size), 'k--', linewidth=linewidth)

    plt.ylim(0, 1.1)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tick_params(axis='both', which='minor', labelsize=labelsize)
    plt.xlabel('Proportion of nodes visited', fontsize=fontsize)
    plt.ylabel('Correct inference of link', fontsize=fontsize)
    plt.show()

    f.savefig("link_inference.pdf", bbox_inches='tight')

    return


def plot_acc_vs_sum_nodes_edges(results, allowed_widths, coos, filt_size, wids, n_available_states, n_available_edges):
    if len(results) > 1:
        f = plt.figure(figsize=(18, 5))
    else:
        f = plt.figure(figsize=(5, 5))
    for i, (res, wid) in enumerate(zip(results, allowed_widths)):
        node_visited_all = res[1]
        edge_visited_all = res[2]

        coos_ = coos[wids == wid]
        coo_mean = np.mean(coos_, 0)

        filt = smooth(coo_mean, filt_size)
        plt.subplot(1, len(np.unique(allowed_widths)), i + 1)
        plt.plot(filt, 'k', alpha=0.4, label='TEM', linewidth=linewidth)

        nodes_visited = np.cumsum(np.mean(np.transpose([x / n_available_states[wids == wid] for x in np.transpose(
            node_visited_all)]), 0))
        edges_visited = np.cumsum(np.mean(np.transpose([x / n_available_edges[wids == wid] for x in np.transpose(
            edge_visited_all)]), 0))

        plt.plot(smooth(nodes_visited, filt_size), 'r--', label='Proportion of nodes visited', linewidth=linewidth)
        plt.plot(smooth(edges_visited, filt_size), 'b--', label='Proportion of edges visited', linewidth=linewidth)

        plt.xlabel('# steps taken', fontsize=fontsize)
        plt.ylabel('Prediction accuracy', fontsize=fontsize)

        plt.ylim([0, 1.1])
        plt.legend(prop={'size': 15})
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        plt.tick_params(axis='both', which='minor', labelsize=labelsize)
    plt.show()

    f.savefig("acc_vs_sum_nodes.pdf", bbox_inches='tight')

    return


def plot_sim_corrects(results, allowed_widths, filt_size, wids, n_available_states, n_available_edges, pars):
    if len(results) > 1:
        f = plt.figure(figsize=(18, 5))
    else:
        f = plt.figure(figsize=(5, 5))
    for i, (res, wid) in enumerate(zip(results, allowed_widths)):
        node_visited_all = res[1]
        edge_visited_all = res[2]

        node_correct_, edge_correct_ = [], []
        for node_visited, edge_visited in zip(node_visited_all, edge_visited_all):
            node_correct, edge_correct = simulate_node_edge_agent(node_visited, edge_visited, acc=pars['acc_simu'],
                                                                  s_size=pars['s_size'])
            node_correct_.append(node_correct)
            edge_correct_.append(edge_correct)

        n_coo_mean = np.mean(node_correct_, 0)
        e_coo_mean = np.mean(edge_correct_, 0)

        plt.subplot(1, len(np.unique(allowed_widths)), i + 1)
        plt.plot(smooth(n_coo_mean, filt_size), 'r', alpha=0.4, label='Node agent', linewidth=linewidth)
        plt.plot(smooth(e_coo_mean, filt_size), 'b', alpha=0.4, label='Edge agent', linewidth=linewidth)

        nodes_visited = np.cumsum(np.mean(np.transpose([x / n_available_states[wids == wid] for x in np.transpose(
            node_visited_all)]), 0))
        edges_visited = np.cumsum(np.mean(np.transpose([x / n_available_edges[wids == wid] for x in np.transpose(
            edge_visited_all)]), 0))

        plt.plot(smooth(nodes_visited, filt_size), 'r--', label='Proportion of nodes visited', linewidth=linewidth)
        plt.plot(smooth(edges_visited, filt_size), 'b--', label='Proportion of edges visited', linewidth=linewidth)

        plt.xlabel('# steps taken', fontsize=fontsize)
        plt.ylabel('Prediction accuracy', fontsize=fontsize)

        plt.ylim([0, 1.1])
        plt.legend(prop={'size': 15})
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        plt.tick_params(axis='both', which='minor', labelsize=labelsize)

    plt.show()

    f.savefig("acc_vs_sum_nodes_simulated.pdf", bbox_inches='tight')

    return


def plot_acc_vs_steps_since_visited(results, allowed_widths, a_s, s_size):

    f = plt.figure()

    edges = [(x+y)/2 for x, y in zip(a_s, a_s[1:])]

    for i, (res, wid) in enumerate(zip(results, allowed_widths)):
        s_s_c = np.asarray(res[3])
        s_s_c = s_s_c[:, :, 0]/s_s_c[:, :, 1]
        mean_ssc = np.nanmean(s_s_c, 0)
        std_ssc = np.nanstd(s_s_c, 0)
        plt.plot(edges, mean_ssc, label=str(wid), linewidth=linewidth)

        plt.fill_between(edges, mean_ssc + std_ssc, mean_ssc - std_ssc,  alpha=0.2)
    plt.xlabel('# steps since visited', fontsize=fontsize)
    plt.ylabel('Prediction accuracy', fontsize=fontsize)
    plt.plot((edges[0], edges[-1]), (1/s_size, 1/s_size), 'k--', linewidth=linewidth)
    plt.ylim(0, 1.1)
    plt.xticks(edges[0:1] + edges[5:-1])

    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tick_params(axis='both', which='minor', labelsize=labelsize)
    plt.show()

    f.savefig("beyond_bptt.pdf", bbox_inches='tight')

    return
