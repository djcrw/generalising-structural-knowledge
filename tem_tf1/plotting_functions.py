#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import scipy.signal as sig
import matplotlib.pyplot as plt
import re
from helper_functions import *
from os import listdir
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from skimage.transform import resize

interpolation_method = 'None'
fontsize = 25
linewidth = 4
labelsize = 20


def im_sq2hex(im_sq, mult=2):
    # mult must be even - round up to nearest even
    mult += np.mod(mult, 2)
    wid = np.shape(im_sq)[0]
    mult_2 = int(mult / 2)
    im_hex = np.zeros((mult * wid, mult * wid + mult_2))

    for row in range(wid):
        for col in range(wid):

            if np.mod(row, 2) == 1:
                col_index = mult * col
            else:
                # shift even rows half to right
                col_index = mult * col + mult_2

            im_hex[mult * row: mult * (row + 1), col_index: col_index + mult] = im_sq[row, col]

    return im_hex


def square_upsample(im_sq, mult=2):
    if mult == 1:
        return im_sq
    else:
        # mult must be even - round up to nearest even
        mult += np.mod(mult, 2)
        height, width = np.shape(im_sq)
        mult_h = mult if height > 1 else 1
        mult_w = mult if width > 1 else 1
        im_up = np.zeros((mult_h * height, mult_w * width))

        for row in range(height):
            for col in range(width):
                im_up[mult_h * row: mult_h * (row + 1), mult_w * col: mult_w * (col + 1)] = im_sq[row, col]

        return im_up


def reshape_cells(cell_, width, world_type):
    if world_type in ['line_ti', 'family_tree', 'loop_laps']:
        cell_reshaped = np.reshape(cell_, (1, -1))
    elif world_type in ['rectangle']:
        height = int(len(cell_) / width)
        cell_reshaped = np.reshape(cell_, (height, width))
    else:
        cell_reshaped = np.reshape(cell_, (width, width))

    return cell_reshaped


def cell_plot_prepare(cell_, width, hexy, smoothing, mult, mask):
    cell_reshaped = reshape_cells(cell_, width, hexy)

    if hexy == 'hex':
        g_im_hex = im_sq2hex(cell_reshaped, mult)
        y_, x_ = np.shape(g_im_hex)
        cell_reshaped = resize(g_im_hex, (int(y_ * np.sqrt(3) / 2), x_))
    else:
        cell_reshaped = square_upsample(cell_reshaped, mult)

    if smoothing:
        kernel = Gaussian2DKernel(x_stddev=smoothing)
        cell_reshaped = convolve(cell_reshaped, kernel)
        if isinstance(mask, np.ndarray):
            cell_reshaped[mask] = np.nan

    if hexy == 'loop_laps':
        cell_reshaped = np.repeat(cell_reshaped, 14, axis=0)

    return cell_reshaped


def square_plot(cell, width, name, maxmin=False, shiny=(), hexy='no', lims=False, smoothing=False, mult=2, cmap='jet',
                mask=False, n_laps=4):
    n = np.shape(cell)[1]  # number of cells we have
    wid = np.ceil(np.sqrt(n))  # dim for subplots
    wid0, wid1 = wid, wid
    if hexy == 'loop_laps':
        wid1 = 3
        wid0 = np.ceil(n / wid1)
    f = plt.figure(figsize=(9, 9))

    for grid in range(n):
        cell_ = cell[:, grid]

        ax = plt.subplot(wid0, wid1, grid + 1)

        cell_reshaped = cell_plot_prepare(cell_, width, hexy, smoothing, mult, mask)

        if lims:
            ax.imshow(cell_reshaped, cmap=cmap, interpolation=interpolation_method, vmin=lims[0][grid],
                      vmax=lims[1][grid])
        else:
            ax.imshow(cell_reshaped, cmap=cmap, interpolation=interpolation_method)

        if shiny is not None:
            shiny = list(shiny)
            for shine in shiny:

                y = int(shine / width)
                x = shine % width
                if hexy == 'hex':
                    x += 0.5 * (1 - np.mod(y, 2))
                y, x = y * mult + (mult - 1) / 2, x * mult + (mult - 1) / 2  # plus 0.5 so center of 'mult square'
                if hexy == 'hex':
                    y *= np.sqrt(3) / 2
                ax.scatter(x, y, c='r', s=10)

        if hexy == 'loop_laps':
            n_states = cell_reshaped.shape[1]

            for iii in range(n_laps):
                # draw line to mark new lap
                ax.plot([n_states * iii / n_laps - 1, n_states * iii / n_laps - 1], [-1, 14], linewidth=2,
                        color='w')

        ax.set_xticks([])
        ax.set_yticks([])

        if maxmin:
            maxi = max(cell_)
            mini = min(cell_)
            ax.set_title("{:.2f},{:.2f}".format(mini, maxi), {'fontsize': 10})
    plt.tight_layout(pad=0.35)
    plt.show()

    f.savefig(name + ".pdf", bbox_inches='tight')

    plt.close('all')


def square_autocorr_plot(cell, width, name, show=True, hexy='no', smoothing=False, mult=2, cmap='jet', circle=False):
    n = np.shape(cell)[1]
    wid = np.ceil(np.sqrt(n))
    f = plt.figure(figsize=(9, 9))
    for grid in range(n):
        ax = plt.subplot(wid, wid, grid + 1)
        cell_ = cell[:, grid]

        cell_reshaped = cell_plot_prepare(cell_, width, hexy, smoothing, mult, False)

        y_, x_ = np.shape(cell_reshaped)

        auto = sig.correlate2d(cell_reshaped, cell_reshaped)

        if circle:
            mask = np.ones_like(auto)
            ys, xs = np.shape(mask)
            radius_sq = (3 / 4) * y_ ** 2
            for y in range(ys):
                for x in range(xs):
                    if (y - y_ + 1) ** 2 + (x - x_ + 1) ** 2 > radius_sq:  # 3/4 for hexagon sides closer at 30 degrees
                        mask[y, x] = np.nan
            auto = auto * mask
            radius = np.sqrt(radius_sq)
            y_indent = int(np.floor(ys / 2) - np.floor(radius)) - 1
            x_indent = int(np.floor(xs / 2) - np.floor(radius)) - 1
            auto = auto[y_indent: -y_indent, x_indent: -x_indent]

        ax.imshow(auto, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(pad=0.15)
    if show:
        plt.show()

    f.savefig(name + ".pdf", bbox_inches='tight')

    plt.close('all')


def get_path(run, date, save_dirs, recent=-1):
    for save_dir in save_dirs:
        try:
            save_path = save_dir + date + '/run' + str(run) + '/save'
            list_of_files = listdir(save_path)
            print('yes' + save_path)

            list_of_files_1 = [x for x in list_of_files if 'link' not in x]

            a = [int(x.split('.')[0].split('_')[-1]) for x in list_of_files_1 if 'par' not in x and '.npy' in x]
            a.sort()

            index = str(np.unique(a)[recent])
            print(index, len(np.unique(a)))

            return save_path, index, list_of_files
        except FileNotFoundError:
            save_path = save_dir + date + '/run' + str(run) + '/save'
            print('not ' + save_path)
            pass

    raise ValueError('FILE NOT FOUND')


def get_data(save_dirs, run, date, recent=-1):
    save_path, index, list_of_files = get_path(run, date, save_dirs, recent)

    a_rnn = np.load(save_path + '/A_RNN_' + index + '.npy')
    g2g = np.load(save_path + '/g2g_' + index + '.npy')
    x_all = np.load(save_path + '/x_all_' + index + '.npy')
    g_all = np.load(save_path + '/g_all_' + index + '.npy')
    p_all = np.load(save_path + '/p_all_' + index + '.npy')
    p_gen_all = np.load(save_path + '/p_gen_all_' + index + '.npy')
    acc_s_t_to = np.load(save_path + '/acc_s_t_to_' + index + '.npy')
    acc_s_t_from = np.load(save_path + '/acc_s_t_from_' + index + '.npy')
    positions = np.load(save_path + '/pos_count_' + index + '.npy')
    shinys = np.load(save_path + '/shiny_states_' + index + '.npy')
    adj = np.load(save_path + '/adj_' + index + '.npy')

    g_all = np.nan_to_num(g_all)
    p_all = np.nan_to_num(p_all)

    pars = np.load(save_path + '/params.npy')
    params = pars.item()

    print(int(index) * params['BPTT_truncation'])
    widths = np.load(save_path + '/widths_' + index + '.npy')

    batch_id = params['diff_env_batches_envs']
    g_size = params['g_size']
    p_size = params['p_size']
    s_size = params['s_size']
    s_size_comp = params['s_size_comp']

    n_freq = params['n_freq']
    width = widths[0]
    states = width ** 2

    g_timeseries = np.load(save_path + '/gs_timeseries_' + index + '.npy')
    try:
        p_timeseries = np.load(save_path + '/ps_timeseries_' + index + '.npy')
    except FileNotFoundError:
        p_timeseries = None
        print('no p_timeseries')
    pos_timeseries = np.load(save_path + '/pos_timeseries_' + index + '.npy')
    timeseries = (g_timeseries, p_timeseries, pos_timeseries)

    data = (a_rnn, g2g, x_all, g_all, p_all, p_gen_all, acc_s_t_to, acc_s_t_from, positions, shinys, adj, timeseries)
    para = (params, widths, batch_id, g_size, p_size, s_size, s_size_comp, n_freq, width, states)

    return data, para, list_of_files, save_path


def sort_data(g_all, p_all, shinys, widths, mult, smoothing, params, batch_id, g_max_0=False, p_max_0=True):
    # sort shiny data_structure
    shinys_ = shinys

    # make masks
    try:
        if params['world_type'] == 'hex':
            masks = make_hex_mask(g_all, widths[batch_id], mult)
        else:
            masks = [False for _ in widths]
    except KeyError:
        params['world_type'] = 'square'
        masks = [False for _ in widths]

    if g_max_0:
        for i, g in enumerate(g_all):
            g_all[i] = np.maximum(g, 0)
    if p_max_0:
        for i, p in enumerate(p_all):
            p_all[i] = np.maximum(p, 0)
    # max / min stuff
    g_smoothed = [[cell_plot_prepare(cell_, widths[batch_id[env]], params['world_type'], smoothing, mult,
                                     masks[env]).flatten() for cell_ in grid.T] for env, grid in enumerate(g_all)]

    p_smoothed = [[cell_plot_prepare(cell_, widths[batch_id[env]], params['world_type'], smoothing, mult,
                                     masks[env]).flatten() for cell_ in grid.T] for env, grid in enumerate(p_all)]

    g_lim = [np.nanmin([np.nanmin(x, 1) for x in g_smoothed], 0), np.nanmax([np.nanmax(x, 1) for x in g_smoothed], 0)]
    p_lim = [np.nanmin([np.nanmin(x, 1) for x in p_smoothed], 0), np.nanmax([np.nanmax(x, 1) for x in p_smoothed], 0)]

    return shinys_, masks, g_lim, p_lim


def extract_number(f):
    s = re.findall("(\d+).npy", f)
    return int(s[0]) if s else -1, f


def make_hex_mask(g_all, widths, mult, hexy='hex'):
    masks = []
    for g, width in zip(g_all, widths):
        cell_ = g[:, 0]
        cell_reshaped = reshape_cells(cell_, width, hexy)
        g_im_hex = im_sq2hex(cell_reshaped, mult)
        y_, x_ = np.shape(g_im_hex)
        cell_reshaped = resize(g_im_hex, (int(y_ * np.sqrt(3) / 2), x_))
        value = cell_reshaped[0, 0]
        mask = cell_reshaped == value

        masks.append(mask)

    return masks


def remove_zero_adj(adj_orig):
    adj = cp.deepcopy(adj_orig)
    for node in reversed(range(len(adj))):
        # if node connects to nothing, or only itself
        if sum(adj[node]) == 0 or (sum(adj[node]) == 1 and adj[node, node] == 1):
            adj = np.delete(adj, node, 0)
            adj = np.delete(adj, node, 1)
    return adj
