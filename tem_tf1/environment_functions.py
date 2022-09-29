#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import numpy as np
import copy as cp


def make_environments(par):
    n_envs = len(par['widths'])
    states_mat = [0] * n_envs
    shiny_states = [0] * n_envs
    n_senses = [par['s_size']] * par['n_freq']
    adjs, trans = [], []

    for env in range(n_envs):

        width = par['widths'][env]
        height = par['heights'][env]

        if par['world_type'] == 'square':
            adj, tran = square_world(width, par['stay_still'])
        elif par['world_type'] == 'rectangle':
            adj, tran = rectangle_world(width, height, par['stay_still'])
        elif par['world_type'] == 'hex':
            hex_boundary = par['hex_boundary']
            adj, tran = hex_world(width, par['stay_still'], hex_boundary)
        elif par['world_type'] == 'family_tree':
            adj, tran = family_tree_world(levels=width)
        elif par['world_type'] == 'line_ti':
            adj, tran = line_ti_world(length=width, jump_length=par['jump_length'][env])
        elif par['world_type'] == 'loop_laps':
            adj, tran = loop_laps_world(width, par['n_laps'], stay_still=False)
        else:
            raise ValueError('incorrect world specified')

        adjs.append(adj)
        trans.append(tran)

        states_mat[env], shiny_states[env] = torus_state_data(n_senses, adj, env, par)

    return adjs, trans, states_mat, shiny_states


def get_new_data_diff_envs(position, data_envs, envs, states_mat, params):
    b_s = int(params['batch_size'])
    n_walk = params['n_walk']
    n_senses = params['n_senses']
    s_size = params['s_size']

    data = np.zeros((b_s, s_size, n_walk + 1))
    for batch in range(b_s):
        env = envs[batch]

        data[batch] = sample_data(position[batch, :], states_mat[env], n_senses, data_envs[batch])

    return data


def sample_data(position, states_mat, n_senses, last_data):

    time_steps = np.shape(position)[0]
    sense_data = np.zeros((n_senses[0], time_steps))
    sense_data[:, 0] = last_data[:, -1]
    for i, pos in enumerate(position):
        if i > 0:
            ind = int(pos)
            sense_data[int(states_mat[ind, 0]), i] = 1
    return sense_data


def get_walking_data(start_state, adj, tran, prev_d, shiny_states, n_walk, params):
    b_s = int(params['batch_size'])

    pos, d = np.zeros((b_s, n_walk + 1)), np.zeros((b_s, params['n_actions'], n_walk))

    for b in range(b_s):
        env = params['diff_env_batches_envs'][b]
        s_s = cp.deepcopy(shiny_states[env])

        if params['world_type'] in ['square', 'rectangle']:
            pos[b, :], d[b, :, :], prev_d[b] = walk_square(adj[env], tran[env], n_walk, start_state[b], prev_d[b], s_s,
                                                           env, params)
        elif params['world_type'] == 'hex':
            pos[b, :], d[b, :, :], prev_d[b] = walk_hex(adj[env], tran[env], n_walk, start_state[b], prev_d[b], s_s,
                                                        env, params)
        elif params['world_type'] == 'family_tree':
            pos[b, :], d[b, :, :], prev_d[b] = walk_family_tree(adj[env], tran[env], n_walk, start_state[b], prev_d[b])
        elif params['world_type'] == 'line_ti':
            pos[b, :], d[b, :, :], prev_d[b] = walk_line_ti(adj[env], tran[env], n_walk, start_state[b], prev_d[b],
                                                            env, params)
        elif params['world_type'] == 'loop_laps':
            pos[b, :], d[b, :, :], prev_d[b] = walk_loop_laps(adj[env], tran[env], n_walk, start_state[b], prev_d[b],
                                                              params)
        else:
            raise ValueError('incorrect world specified')

    return pos, d


def curriculum(pars_orig, pars, n_restart):
    n_envs = len(pars['widths'])
    b_s = int(pars['batch_size'])
    # choose pars for current stage of training
    # choose between shiny / normal

    rn = np.random.randint(low=-pars['seq_jitter'], high=pars['seq_jitter'])
    n_restart = np.maximum(n_restart - pars['curriculum_steps'], pars['restart_min'])

    pars['shiny_bias_env'] = [(0, 0) for _ in range(n_envs)]
    pars['direc_bias_env'] = [0 for _ in range(n_envs)]

    pars['shiny_sense'], shiny_s = choose_shiny_sense(pars)

    # make choice for each env
    choices = []
    for env in range(n_envs):
        choice = np.random.choice(pars['poss_behaviours'])

        choices.append(choice)

        if choice == 'shiny':
            pars['shiny_bias_env'][env] = pars_orig['shiny_bias']
        elif choice == 'normal':
            pars['direc_bias_env'][env] = pars_orig['direc_bias']
        else:
            raise Exception('Not a correct possible behaviour')

    # shiny_s for each batch
    for batch in range(b_s):
        env = pars['diff_env_batches_envs'][batch]
        choice = choices[env]
        if choice == 'normal':
            shiny_s[batch, :] = 0

    # choose which of batch gets no_direc or not - 1 is no_direc, 0 is with direc
    no_direc_batch = np.ones(pars['batch_size'])
    for batch in range(b_s):
        env = pars['diff_env_batches_envs'][batch]
        choice = choices[env]
        if choice == 'normal':
            no_direc_batch[batch] = 0
        else:
            no_direc_batch[batch] = 1

    return pars, shiny_s, rn, n_restart, no_direc_batch


def torus_state_data(n_senses, adj, env, par):
    width = par['widths'][env]
    shiny_bias = par['shiny_bias_env'][env]
    shiny_sense = par['shiny_sense'][env]
    n_states = par['n_states_world'][env]

    n_freq = np.size(n_senses)
    states_vec = np.zeros((n_states, 1))
    shiny_use = True if shiny_bias[0] > 0 else False
    choices = np.arange(n_senses[0])
    shiny_states = None

    if shiny_use:
        max_sep = np.maximum((width - 2) / len(shiny_sense), 4)

        shiny_states = []
        while len(shiny_states) < 2:
            shiny_states = []
            # choose shiny state so not on boundary
            allowed_states = [x for x in range(n_states) if np.sum(adj, 0)[x] == np.max(np.sum(adj, 0))]

            for i in range(len(shiny_sense)):
                # choose shiny position
                s_s = np.random.choice(allowed_states)
                shiny_states.append(s_s)
                # update allowed states given shiny position
                if i < len(shiny_sense) - 1:
                    allowed_states = [x for x in allowed_states if
                                      np.min(distance_between_states(x, s_s, width, par['world_type'])) > max_sep]

                if not allowed_states:
                    print('No space to put object ' + str(i + 2), shiny_states)
                    break
            max_sep += -0.5  # reduce max_sep if cant find space to put at least 2 shinies in each room
        print(max_sep + 0.5, len(shiny_states))

    if shiny_use:
        # remove shiny senses from available sense
        shiny_sense_sorted = sorted(list(set(shiny_sense)), reverse=True)
        for s_s in shiny_sense_sorted:
            # this requires choices be ordered + sense not repeated (hence set)
            choices = np.delete(choices, s_s)

    if par['world_type'] in ['loop_laps']:
        # choose reward sense
        reward_sense = np.random.choice(choices)
        # choices = np.delete(choices, reward_sense)
    else:
        reward_sense = 0

    for i in range(n_states):
        if par['world_type'] == 'loop_laps':
            new_state = np.random.choice(choices)
            len_loop = int(n_states / par['n_laps'])

            states_vec[i, 0] = new_state if i / len_loop < 1 else states_vec[i - len_loop, 0]

        else:
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i, 0] = new_state

    if par['world_type'] in ['loop_laps']:
        # make particular position special in track
        states_vec[par['reward_pos'], 0] = reward_sense

    if shiny_use:
        # put shinies in state_mat
        for sense, state in zip(shiny_sense, shiny_states):
            # assign sense to states
            states_vec[state, 0] = sense

    states_mat = np.repeat(states_vec, n_freq, axis=1)
    return states_mat, shiny_states


def square_world(width, stay_still):
    """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
    """
    states = int(width ** 2)
    adj = np.zeros((states, states))

    for i in range(states):
        # stay still
        if stay_still:
            adj[i, i] = 1
        # up - down
        if i + width < states:
            adj[i, i + width] = 1
            adj[i + width, i] = 1
            # left - right
        if np.mod(i, width) != 0:
            adj[i, i - 1] = 1
            adj[i - 1, i] = 1

    tran = np.zeros((states, states))
    for i in range(states):
        if sum(adj[i]) > 0:
            tran[i] = adj[i] / sum(adj[i])

    return adj, tran


def rectangle_world(width, height, stay_still):
    """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
    """
    states = int(width * height)
    adj = np.zeros((states, states))

    for i in range(states):
        # stay still
        if stay_still:
            adj[i, i] = 1
        # up - down
        if i + width < states:
            adj[i, i + width] = 1
            adj[i + width, i] = 1
            # left - right
        if np.mod(i, width) != 0:
            adj[i, i - 1] = 1
            adj[i - 1, i] = 1

    tran = np.zeros((states, states))
    for i in range(states):
        if sum(adj[i]) > 0:
            tran[i] = adj[i] / sum(adj[i])

    return adj, tran


def hex_world(width, stay_still, hex_boundary=False):
    """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
    """
    states = int(width ** 2)
    adj_box = np.zeros((states, states))

    for n1 in range(width):  # row
        for n2 in range(width):  # col
            if n2 < width - 1:
                # go right
                adj_box[n1 * width + n2, n1 * width + n2 + 1] = 1
            if n1 < width - 1:
                # go down
                adj_box[n1 * width + n2, (n1 + 1) * width + n2] = 1
                if np.mod(n1, 2) == 0 and n2 < width - 1:
                    # go down and right
                    adj_box[n1 * width + n2, (n1 + 1) * width + n2 + 1] = 1
                elif np.mod(n1, 2) == 1 and n2 > 0:
                    # go down and left
                    adj_box[n1 * width + n2, (n1 + 1) * width + n2 - 1] = 1

            if stay_still:
                adj_box[n1 * width + n2, n1 * width + n2] = 1
    adj_box = adj_box + adj_box.T
    adj_box = np.minimum(adj_box, 1)

    if hex_boundary:
        adj_box = square2hex(adj_box)

    tran_box = np.zeros((states, states))
    for i in range(states):
        if sum(adj_box[i]) > 0:
            tran_box[i] = adj_box[i] / sum(adj_box[i])

    return adj_box, tran_box


def family_tree_world(levels=3):
    # each person has 2 children
    """
               0
          /        \
         1          2
        / \        /  \
       3   4      5    6
      / \ / \    / \  / \
      7 8 9 10  11 12 13 14
    """
    # number of nodes = 2**0 + 2**1 + 2**3 + ... 2**levels = 2**(levels+1) - 1
    n_nodes = 2 ** (levels + 1) - 1

    adj = np.zeros((n_nodes, n_nodes))

    # children
    for i in range(n_nodes - 2 ** levels):
        # child 1
        adj[i, 2 * (i + 1) - 1] = 1
        # child 2
        adj[i, 2 * (i + 1)] = 1
    # parents
    adj += adj.T  # or np.floor((i-1)/2)

    for i in range(n_nodes):
        # siblings
        if i % 2 == 1:
            adj[i, i + 1] = 1
            adj[i + 1, i] = 1

        # grand parent
        g_p_i = int(np.floor((np.floor((i - 1) / 2) - 1) / 2))
        if g_p_i >= 0:
            adj[i, g_p_i] = 1

        p_i = int((np.floor((i - 1) / 2)))
        if p_i > 0:
            # uncle/aunt
            if p_i % 2 == 1:
                u_i = p_i + 1
            else:
                u_i = p_i - 1
            adj[i, u_i] = 1

            # niece/nephew
            adj[u_i, i] = 1

            # cousins 1
            adj[i, 2 * (u_i + 1) - 1] = 1

            # cousin 2
            adj[i, 2 * (u_i + 1)] = 1

    tran = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        if sum(adj[i]) > 0:
            tran[i] = adj[i] / sum(adj[i])

    return adj, tran


def line_ti_world(length=10, jump_length=9):
    adj = np.zeros((length, length))

    for row in range(length):
        for col in range(length):
            diff = col - row
            if np.abs(diff) <= jump_length and diff != 0:
                adj[row, col] = 1

    tran = np.zeros((length, length))
    for i in range(length):
        if sum(adj[i]) > 0:
            tran[i] = adj[i] / sum(adj[i])

    return adj, tran


def loop_laps_world(width, n_laps, stay_still=False):
    n_states = n_laps * (2 * width + 2 * (width - 2))

    adj = np.zeros((n_states, n_states))

    # go round track twice
    for i in range(n_states):
        if i < n_states - 1:
            adj[i, i + 1] = 1

        if stay_still:
            adj[i, i] = 1

    # lap to beginning:
    adj[n_states - 1, 0] = 1

    tran = np.zeros((n_states, n_states))
    for i in range(n_states):
        if sum(adj[i]) > 0:
            tran[i] = adj[i] / sum(adj[i])

    return adj, tran


def rectangle_relation(s1, s2, width, height):
    # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
    diff = s2 - s1
    if diff == width or diff == -width * (height - 1):  # down
        direc = 0
        name = 'down'
    elif diff == -width or diff == width * (height - 1):  # up
        direc = 1
        name = 'up'
    elif diff == -1 or diff == (width - 1):  # left
        direc = 2
        name = 'left'
    elif diff == 1 or diff == -(width - 1):  # right
        direc = 3
        name = 'right'
    elif diff == 0:
        direc = 4
        name = 'stay still'
    else:
        raise ValueError('impossible action')

    return direc, name


def hex_relation(s1, s2, width):
    level = np.mod(int(s1 / width), 2)
    diff = s2 - s1
    if diff == width - 1 or (diff == width and level == 0):  # down left
        direc = 0
    elif diff == width + 1 or (diff == width and level == 1):  # down right
        direc = 1
    elif diff == -(width + 1) or (diff == -width and level == 0):  # up left
        direc = 2
    elif diff == -(width - 1) or (diff == -width and level == 1):  # up right
        direc = 3
    elif diff == -1:  # left
        direc = 4
    elif diff == 1:  # right
        direc = 5
    elif diff == 0:
        direc = 6
    else:
        raise ValueError('impossible action')

    return direc


def family_relation_type(s1, s2):
    # find level:
    level_1 = np.floor(np.log2(s1 + 1))
    level_2 = np.floor(np.log2(s2 + 1))

    diff = s2 - s1
    diff_level = level_2 - level_1

    rel_type = 'fail'

    if s2 == np.floor((s1 - 1) / 2):
        rel_type = 'parent'
    elif s2 == 2 * (s1 + 1) - 1:
        rel_type = 'child 1'
    elif s2 == 2 * (s1 + 1):
        rel_type = 'child 2'
    elif diff_level == 0:
        if s1 % 2 == 1:

            if diff == 1:
                rel_type = 'sibling'
            if np.abs(diff) == 2:
                rel_type = 'cousin 1'
            if diff == 3:
                rel_type = 'cousin 2'
            if diff == -1:
                rel_type = 'cousin 2'

        if s1 % 2 == 0:

            if diff == -1:
                rel_type = 'sibling'
            if diff == 1:
                rel_type = 'cousin 1'
            if np.abs(diff) == 2:
                rel_type = 'cousin 2'
            if diff == -3:
                rel_type = 'cousin 1'
    elif diff_level == -2:
        rel_type = 'grand parent'
    elif diff_level == -1:
        rel_type = 'uncle/aunt'
    elif diff_level == 1:
        if s1 % 2 == 0:
            if s2 == 2 * ((s1 - 1) + 1) - 1:
                rel_type = 'niece/nephew 1'
            elif s2 == 2 * ((s1 - 1) + 1):
                rel_type = 'niece/nephew 2'
        else:
            if s2 == 2 * ((s1 + 1) + 1) - 1:
                rel_type = 'niece/nephew 1'
            elif s2 == 2 * ((s1 + 1) + 1):
                rel_type = 'niece/nephew 2'
    else:
        raise ValueError('impossible action')

    rels = ['parent', 'child 1', 'child 2', 'sibling', 'grand parent', 'uncle/aunt', 'niece/nephew 1', 'niece/nephew 2',
            'cousin 1', 'cousin 2']

    rel_index = rels.index(rel_type)

    return rel_type, rel_index


def line_ti_relation(s1, s2, jump_length):
    diff = s2 - s1

    direc = np.sign(diff)
    num = np.abs(diff)

    direc_exact = diff + jump_length

    return direc, num, direc_exact


def loop_laps_relation_type(s1, s2, width, n_laps):
    n_states = n_laps * (2 * width + 2 * (width - 2))
    pos_lap_1 = s1 % int(n_states / n_laps)
    pos_lap_2 = s2 % int(n_states / n_laps)

    if s1 > n_states or s2 > n_states:
        raise ValueError('impossible state index - too high')
    if pos_lap_2 - pos_lap_1 == 0:
        relation = 'stay still'
        direc = 4
    elif s2 - s1 != 1 and s2 - s1 != -(n_states - 1):
        raise ValueError('impossible state transition')
    elif 0 < pos_lap_2 < width:
        relation = 'right'
        direc = 3
    elif width <= pos_lap_2 < width + 1 * (width - 1):
        relation = 'up'
        direc = 1
    elif width + 1 * (width - 1) <= pos_lap_2 < width + 2 * (width - 1):
        relation = 'left'
        direc = 2
    elif pos_lap_2 < width + 3 * (width - 1):
        relation = 'down'
        direc = 0
    else:
        raise ValueError('impossible action')

    return relation, direc


def walk_square(adj, tran, time_steps, start_state, prev_dir, shiny_state, env, params):
    """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
    """
    position = np.zeros((1, time_steps + 1), dtype=np.int16)
    direc = np.zeros((4, time_steps))
    if params['world_type'] == 'rectangle':
        width = params['widths'][env]
    else:
        width = int(np.sqrt(np.shape(adj)[0]))

    shiny_bias = params['shiny_bias_env'][env]

    sb_min = shiny_bias[0]
    sb_max = shiny_bias[1]

    shiny_b, rn, shiny_s_ind, shiny_recent, ps_current = None, None, None, None, None
    current_angle = np.random.uniform(-np.pi, np.pi)

    # consider rectangular environment. if move +1=up, -1=down, +length=right, -length=left.
    if params['world_type'] == 'rectangle':
        height, wid = params['heights'][env], params['widths'][env]
        if height * wid != len(adj):
            raise ValueError('incorrect heigh/width : height * width not equal to number of states')
    else:
        height, wid = width, width

    distance_index = 1  # euclidean=0 , steps=1
    if sb_min > 0:
        # object want to go to
        shiny_s_ind = np.random.choice(np.arange(len(shiny_state)))
        shiny_b = [shiny_s_ind, shiny_state[shiny_s_ind], 0]
        rn = np.random.randint(params['object_hang_min'], params['object_hang_max'])

    position[0, 0] = int(start_state)
    for i in range(time_steps):
        available = np.where(tran[int(position[0, i]), :] > 0)[0].astype(int)

        # head towards objects, or in straight lines
        if sb_min > 0:
            # bias towards objects
            # choose new object to go to
            if shiny_b[2] > rn:
                try:
                    shiny_s_ind = np.random.choice([x for x in range(len(shiny_state)) if x != shiny_s_ind])
                except ValueError:
                    shiny_s_ind = np.random.choice([x for x in range(len(shiny_state))])
                shiny_b = [shiny_s_ind, shiny_state[shiny_s_ind], 0]
                rn = np.random.randint(params['object_hang_min'], params['object_hang_max'])
            # visited current shiny
            if position[0, i] == shiny_b[1]:
                shiny_b[2] += 1

            distances = [distance_between_states(shiny_state[shiny_s_ind], x, width, params['world_type'])
                         [distance_index] for x in available]

            ps = [1 / x for x in distances]
            ps = [x / sum(ps) for x in ps]

            # when in vicinity of object, move there more voraciously.
            # i.e. when not in vicinity this allows better exploration

            # bias to current object of choice
            g = np.zeros_like(available).astype(np.float32)
            min_dis_ind = np.random.choice(np.where(distances == min(distances))[0])
            g[min_dis_ind] = 1

            p = (sb_min * g) + (1 - sb_min - sb_max) * tran[int(position[0, i]), available] + sb_max * np.asarray(ps)

            # Staying still should always occur a certain proportion of time of time
            stay_still_pos = np.where(available == int(position[0, i]))[0]
            if len(stay_still_pos) > 0:
                p = (1 - params['object_stay_still']) * p / sum(p[np.arange(len(p)) != stay_still_pos[0]])
                p[stay_still_pos[0]] = params['object_stay_still']
            new_poss_pos = np.random.choice(available, p=p)
        elif params['bias_type'] == 'angle':
            new_poss_pos, current_angle = move_straight_bias(current_angle, position[0, i], width, available, tran,
                                                             params)
        else:
            new_poss_pos = np.random.choice(available)

        if adj[position[0, i], new_poss_pos] == 1:
            position[0, i + 1] = new_poss_pos
        else:
            position[0, i + 1] = int(cp.deepcopy(position[0, i]))

        prev_dir, _ = rectangle_relation(position[0, i], position[0, i + 1], wid, height)
        if prev_dir < 4:
            direc[prev_dir, i] = 1
        # stay still is just a set of zeros

    return position, direc, prev_dir


def walk_hex(adj, tran, time_steps, start_state, prev_dir, shiny_state, env, params):
    """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
    """
    position = np.zeros((1, time_steps + 1), dtype=np.int16)
    direc = np.zeros((6, time_steps))
    width = int(np.sqrt(np.shape(adj)[0]))

    shiny_bias = params['shiny_bias_env'][env]

    sb_min = shiny_bias[0]
    sb_max = shiny_bias[1]

    shiny_b, rn, shiny_s_ind, shiny_recent, ps_current = None, None, None, None, None
    current_angle = np.random.uniform(-np.pi, np.pi)

    distance_index = 0  # euclidean=0 , steps=1
    if sb_min > 0:
        # methods of shiny biasing
        # object want to go to
        shiny_s_ind = np.random.choice(np.arange(len(shiny_state)))
        shiny_b = [shiny_s_ind, shiny_state[shiny_s_ind], 0]
        rn = np.random.randint(params['object_hang_min'], params['object_hang_max'])

    position[0, 0] = int(start_state)
    for i in range(time_steps):
        available = np.where(tran[int(position[0, i]), :] > 0)[0].astype(int)

        # bias towards shiny state
        if sb_min > 0:
            distances = [[distance_between_states(s_s, x, width, params['world_type'])[distance_index]
                          for x in available] for s_s in shiny_state]  # steps

            # choose new object to go to
            if shiny_b[2] > rn:
                shiny_s_ind = np.random.choice([x for x in range(len(shiny_state)) if x != shiny_s_ind])
                shiny_b = [shiny_s_ind, shiny_state[shiny_s_ind], 0]
                rn = np.random.randint(params['object_hang_min'], params['object_hang_max'])
            # visited current shiny
            if position[0, i] == shiny_b[1]:
                shiny_b[2] += 1

            distances_current = [distance_between_states(shiny_state[shiny_s_ind], x, width,
                                                         params['world_type'])[distance_index]
                                 for x in available]
            ps = [1 / x for x in distances_current]
            ps = [x / sum(ps) for x in ps]

            # when in vicinity of object, move there more voraciously.
            # i.e. when not in vicinity this allows better exploration

            # bias to currently object of choice
            g = np.zeros_like(available).astype(np.float32)
            min_dis_ind = np.random.choice(np.where(distances[shiny_b[0]] == min(distances[shiny_b[0]]))[0])
            g[min_dis_ind] = 1

            p = (sb_min * g) + (1 - sb_min - sb_max) * tran[int(position[0, i]), available] + sb_max * np.asarray(ps)

            # Staying still should always occur a certain proportion of time of time
            stay_still_pos = np.where(available == int(position[0, i]))[0]
            if len(stay_still_pos) > 0:
                p = (1 - params['object_stay_still']) * p / sum(p[np.arange(len(p)) != stay_still_pos[0]])
                p[stay_still_pos[0]] = params['object_stay_still']
            new_poss_pos = np.random.choice(available, p=p)

        elif params['bias_type'] == 'angle':
            new_poss_pos, current_angle = move_straight_bias(current_angle, position[0, i], width, available, tran,
                                                             params)
        else:
            new_poss_pos = np.random.choice(available)

        if adj[position[0, i], new_poss_pos] == 1:
            position[0, i + 1] = new_poss_pos
        else:
            position[0, i + 1] = int(cp.deepcopy(position[0, i]))

        prev_dir = hex_relation(position[0, i], position[0, i + 1], width)
        if prev_dir < 6:
            direc[prev_dir, i] = 1
        # stay still is just a set of zeros

    return position, direc, prev_dir


def walk_family_tree(adj, tran, time_steps, start_state, prev_dir):
    position = np.zeros((1, time_steps + 1), dtype=np.int16)
    direc = np.zeros((10, time_steps))

    position[0, 0] = int(start_state)
    for i in range(time_steps):
        available = np.where(tran[int(position[0, i]), :] > 0)[0].astype(int)
        p = tran[int(position[0, i]), available]  # choose next position from actual allowed positions
        new_poss_pos = np.random.choice(available, p=p)

        if adj[position[0, i], new_poss_pos] == 1:
            position[0, i + 1] = new_poss_pos
        else:
            position[0, i + 1] = int(cp.deepcopy(position[0, i]))

        rel_type, rel_index = family_relation_type(position[0, i], position[0, i + 1])

        direc[rel_index, i] = 1
        prev_dir = rel_index

    return position, direc, prev_dir


def walk_line_ti(adj, tran, time_steps, start_state, prev_dir, env, params):
    position = np.zeros((1, time_steps + 1), dtype=np.int16)
    direc = np.zeros((2, time_steps))

    position[0, 0] = int(start_state)
    for i in range(time_steps):
        available = np.where(tran[int(position[0, i]), :] > 0)[0].astype(int)
        p = tran[int(position[0, i]), available]  # choose next position from actual allowed positions
        new_poss_pos = np.random.choice(available, p=p)

        if adj[position[0, i], new_poss_pos] == 1:
            position[0, i + 1] = new_poss_pos
        else:
            position[0, i + 1] = int(cp.deepcopy(position[0, i]))

        d_, num_, direc_exact_ = line_ti_relation(position[0, i], position[0, i + 1], params['jump_length'][env])

        direc[0, i] = d_
        direc[1, i] = num_
        prev_dir = direc_exact_

    return position, direc, prev_dir


def walk_loop_laps(adj, tran, time_steps, start_state, prev_dir, params):
    position = np.zeros((1, time_steps + 1), dtype=np.int16)
    direc = np.zeros((4, time_steps))
    n_states = np.shape(adj)[0]
    width = int((n_states / params['n_laps'] + 4) / 4)

    position[0, 0] = int(start_state)
    for i in range(time_steps):
        available = np.where(tran[int(position[0, i]), :] > 0)[0].astype(int)
        p = tran[int(position[0, i]), available]  # choose next position from actual allowed positions
        new_poss_pos = np.random.choice(available, p=p)

        if adj[position[0, i], new_poss_pos] == 1:
            position[0, i + 1] = new_poss_pos
        else:
            position[0, i + 1] = int(cp.deepcopy(position[0, i]))

        rel_type, rel_index = loop_laps_relation_type(position[0, i], position[0, i + 1], width, params['n_laps'])

        direc[rel_index, i] = 1
        prev_dir = rel_index

    return position, direc, prev_dir


def move_straight_bias(current_angle, position, width, available, tran, params):
    # angle is allo-centric
    # from available position - find distance and angle from current pos
    if params['world_type'] in ['square', 'rectangle']:
        angle_checker = angle_between_states_square
        diff_angle_min = np.pi / 4
    else:
        angle_checker = angle_between_states_hex
        diff_angle_min = np.pi / 6
    angles = [angle_checker(position, x, width) if x != position else 10000 for x in available]
    # find angle closest to current angle
    a_diffs = [np.abs(a - current_angle) for a in angles]
    a_diffs = [a if a < np.pi else np.abs(2 * np.pi - a) for a in a_diffs]

    angle_diff = np.min(a_diffs)

    if angle_diff < diff_angle_min:
        a_min_index = np.where(a_diffs == angle_diff)[0][0]
        angle = current_angle
    else:  # hit a wall - then do random non stationary choice
        p_angles = [1 if a < 100 else 0.000001 for a in angles]
        a_min_index = np.random.choice(np.arange(len(available)), p=np.asarray(p_angles) / sum(p_angles))
        angle = angles[a_min_index]

    new_poss_pos = int(available[a_min_index])

    angle += np.random.uniform(-params['angle_bias_change'], params['angle_bias_change'])
    angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi  # keep between +- pi

    if np.random.rand() > params['direc_bias']:
        p = tran[int(position), available]
        new_poss_pos = np.random.choice(available, p=p)

    return new_poss_pos, angle


def angle_between_states_square(s1, s2, width):
    x1 = s1 % width
    x2 = s2 % width

    y1 = np.floor(s1 / width)
    y2 = np.floor(s2 / width)

    angle = np.arctan2(y1 - y2, x2 - x1)

    return angle


def angle_between_states_hex(s1, s2, width):
    level_1 = np.mod(int(s1 / width), 2)
    level_2 = np.mod(int(s2 / width), 2)

    x1 = s1 % width
    x2 = s2 % width

    y1 = np.floor(s1 / width)
    y2 = np.floor(s2 / width)

    if level_1 == level_2:
        angle = np.arctan2(y1 - y2, x2 - x1)
    elif level_1 == 0 and level_2 == 1:
        angle = np.arctan2((np.sqrt(3) / 2) * (y1 - y2), x2 - 0.5 - x1)
    elif level_1 == 1 and level_2 == 0:
        angle = np.arctan2((np.sqrt(3) / 2) * (y1 - y2), x2 - x1 + 0.5)
    else:
        raise ValueError('something wrong!!')

    return angle


def distance_between_states(s1, s2, width, world_type):
    x1 = s1 % width
    x2 = s2 % width

    y1 = np.floor(s1 / width)
    y2 = np.floor(s2 / width)

    if world_type == 'hex':
        level_1 = np.mod(y1, 2)
        level_2 = np.mod(y2, 2)

        x1 += -level_1 * 0.5
        x2 += -level_2 * 0.5
        y1 *= np.sqrt(3) / 2
        y2 *= np.sqrt(3) / 2

    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) + 1e-6
    steps = np.abs(x1 - x2) + np.abs(y1 - y2) + 1e-6
    return distance, steps


def choose_shiny_sense(pars):

    # choose number of shiny objects per environment + choose which sensory stimuli will be shiny
    shiny_sense = [np.random.randint(0, pars['n_shiny_senses'], np.random.choice(pars['n_shiny']))
                   for _ in pars['widths']]

    # make mask for model - different for each batch
    shiny_s = np.zeros((pars['batch_size'], pars['s_size']))
    for i, s_s_env in enumerate(shiny_sense):
        for j, s_s_ in enumerate(s_s_env):
            shiny_s[pars['diff_env_batches_envs'][i], s_s_] = 1

    return shiny_sense, shiny_s


def square2hex(a):
    # length must be odd
    n_states = len(a)
    length = int(np.sqrt(len(a)))
    hex_length = (length + 1) / 2

    middle = int((n_states - 1) / 2)
    init = np.zeros(n_states)
    init[middle] = 1

    n_hops = int(hex_length - 1)
    jumps = [init]
    for i in range(n_hops):
        jumps.append(np.dot(a, jumps[i]))

    jumps_add = np.sum(jumps, 0)

    a_new = cp.deepcopy(a)
    for i, val in enumerate(list(jumps_add)):
        if val == 0:
            a_new[i, :] = 0
            a_new[:, i] = 0

    return a_new
