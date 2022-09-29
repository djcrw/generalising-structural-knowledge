#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import numpy as np
import copy as cp
import itertools


class Environment:
    def __init__(self, params, width, height, n_states):
        super(Environment, self).__init__()
        self.par = params
        self.width = width
        self.height = height
        self.n_actions = self.par.env.n_actions
        self.rels = self.par.env.rels
        self.walk_len = None
        self.reward_value = 1.0
        self.reward_pos_training = []
        self.start_state, self.adj, self.tran, self.states_mat = None, None, None, None

        if n_states > self.par.max_states:
            raise ValueError(
                ('Too many states in your world. {} is bigger than {}. Adjust by decreasing environment size, or' +
                 'increasing params.max_states').format(n_states, self.par.max_states))


class Rectangle(Environment):

    def __init__(self, params, width, height):
        self.n_states = width * height

        super().__init__(params, width, height, self.n_states)

    def world(self, torus=False):
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
        states = int(self.width * self.height)
        adj = np.zeros((states, states))

        for i in range(states):
            # stay still
            if self.par.env.stay_still:
                adj[i, i] = 1
            # up - down
            if i + self.width < states:
                adj[i, i + self.width] = 1
                adj[i + self.width, i] = 1
                # left - right
            if np.mod(i, self.width) != 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1

            if torus and np.mod(i, self.width) == 0:
                adj[i, i + self.width - 1] = 1
                adj[i + self.width - 1, i] = 1

            if torus and int(i / self.width) == 0:
                adj[i, i + states - self.width] = 1
                adj[i + states - self.width, i] = 1

        tran = np.zeros((states, states))
        for i in range(states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran
        allowed_states = np.where(np.sum(self.adj, 1) > 0)[0]
        self.start_state = np.random.choice(allowed_states)

    def relation(self, s1, s2):
        # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
        diff = s2 - s1
        if diff == self.width or diff == -self.width * (self.height - 1):  # down
            rel_type = 'down'
        elif diff == -self.width or diff == self.width * (self.height - 1):  # up
            rel_type = 'up'
        elif diff == -1 or diff == (self.width - 1):  # left
            rel_type = 'left'
        elif diff == 1 or diff == -(self.width - 1):  # right
            rel_type = 'right'
        elif diff == 0:
            rel_type = 'stay still'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = np.arange(self.par.s_size)

        for i in range(self.n_states):
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i] = new_state

        self.states_mat = states_vec.astype(int)

    def walk(self):
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
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))

        current_angle = np.random.uniform(-np.pi, np.pi)

        # consider rectangular environment. if move +1=up, -1=down, +length=right, -length=left.
        if self.height * self.width != len(self.adj):
            raise ValueError('incorrect height/width : height * width not equal to number of states')

        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1

        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)

            # head towards objects, or in straight lines
            if self.par.env.bias_type == 'angle':
                new_poss_pos, current_angle = self.move_straight_bias(current_angle, position[i], available)
            else:
                new_poss_pos = np.random.choice(available)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            relation_taken, _ = self.relation(position[i], position[i + 1])
            if relation_taken < self.n_actions:
                direc[relation_taken, i + 1] = 1
            # stay still is just a set of zeros

        return position, direc

    def move_straight_bias(self, current_angle, position, available):
        # angle is allo-centric
        # from available position - find distance and angle from current pos
        diff_angle_min = np.pi / 4
        angles = [self.angle_between_states_square(position, x) if x != position else 10000 for x in available]
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

        angle += np.random.uniform(-self.par.env.angle_bias_change, self.par.env.angle_bias_change)
        angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi  # keep between +- pi

        if np.random.rand() > self.par.env.direc_bias:
            p = self.tran[int(position), available]
            new_poss_pos = np.random.choice(available, p=p)

        return new_poss_pos, angle

    def angle_between_states_square(self, s1, s2):
        x1 = s1 % self.width
        x2 = s2 % self.width

        y1 = np.floor(s1 / self.width)
        y2 = np.floor(s2 / self.width)

        angle = np.arctan2(y1 - y2, x2 - x1)

        return angle

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):

        xs, ys = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xs = xs.flatten() - (self.width - 1) / 2
        ys = - (ys.flatten() - (self.height - 1) / 2)

        if cells is not None:
            cell_prepared = cp.deepcopy(cells).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


class Hexagonal(Environment):
    def __init__(self, params, width):
        self.n_states = int((3 * (width ** 2) + 1) / 4)
        self.graph_states = width ** 2

        super().__init__(params, width, width, self.n_states)

    def world(self):
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
        width = self.width
        states = int(width ** 2)
        adj = np.zeros((states, states))

        for n1 in range(width):  # row
            for n2 in range(width):  # col
                if n2 < width - 1:
                    # go right
                    adj[n1 * width + n2, n1 * width + n2 + 1] = 1
                if n1 < width - 1:
                    # go down
                    adj[n1 * width + n2, (n1 + 1) * width + n2] = 1
                    if np.mod(n1, 2) == 0 and n2 < width - 1:
                        # go down and right
                        adj[n1 * width + n2, (n1 + 1) * width + n2 + 1] = 1
                    elif np.mod(n1, 2) == 1 and n2 > 0:
                        # go down and left
                        adj[n1 * width + n2, (n1 + 1) * width + n2 - 1] = 1

                if self.par.env.stay_still:
                    adj[n1 * width + n2, n1 * width + n2] = 1
        adj = adj + adj.T
        adj = np.minimum(adj, 1)

        if self.par.env.hex_boundary:
            adj = square2hex(adj)

        tran = np.zeros((states, states))
        for i in range(states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran
        allowed_states = np.where(np.sum(self.adj, 1) > 0)[0]
        self.start_state = np.random.choice(allowed_states)

    def relation(self, s1, s2):
        width = self.width
        level = np.mod(int(s1 / width), 2)
        diff = s2 - s1
        if diff == width - 1 or (diff == width and level == 0):
            rel_type = 'down left'
        elif diff == width + 1 or (diff == width and level == 1):
            rel_type = 'down right'
        elif diff == -(width + 1) or (diff == -width and level == 0):
            rel_type = 'up left'
        elif diff == -(width - 1) or (diff == -width and level == 1):
            rel_type = 'up right'
        elif diff == -1:
            rel_type = 'left'
        elif diff == 1:
            rel_type = 'right'
        elif diff == 0:
            rel_type = 'stay still'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.graph_states)
        choices = np.arange(self.par.s_size)

        for i in range(self.graph_states):
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i] = new_state

        self.states_mat = states_vec.astype(int)

    def walk(self):
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))

        current_angle = np.random.uniform(-np.pi, np.pi)
        # breakpoint()
        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1

        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)

            if self.par.env.bias_type == 'angle':
                new_poss_pos, current_angle = self.move_straight_bias(current_angle, position[i], available)
            else:
                new_poss_pos = np.random.choice(available)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            relation_taken, _ = self.relation(position[i], position[i + 1])
            if relation_taken < self.n_actions:
                direc[relation_taken, i + 1] = 1
            # stay still is just a set of zeros

        return position, direc

    def move_straight_bias(self, current_angle, position, available):
        # angle is allo-centric
        # from available position - find distance and angle from current pos
        diff_angle_min = np.pi / 6
        angles = [self.angle_between_states_hex(position, x) if x != position else 10000 for x in available]
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

        angle += np.random.uniform(-self.par.env.angle_bias_change, self.par.env.angle_bias_change)
        angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi  # keep between +- pi

        if np.random.rand() > self.par.env.direc_bias:
            p = self.tran[int(position), available]
            new_poss_pos = np.random.choice(available, p=p)

        return new_poss_pos, angle

    def angle_between_states_hex(self, s1, s2):
        width = self.width
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

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        width = self.width

        xs, ys = np.meshgrid(np.arange(width), np.arange(width))
        xs = (xs.flatten() - (width - 1) / 2) - np.mod(ys.flatten(), 2) * 0.5
        ys = - (ys.flatten() - (width - 1) / 2) * np.sqrt(3) / 2

        if cells is not None:
            cell_prepared = np.asarray(cells).flatten()
            cell_prepared[~in_hexagon(xs, ys, width)] = np.nan

            return xs, ys, cell_prepared
        else:
            return xs, ys


class FamilyTree(Environment):
    def __init__(self, params, levels):
        self.levels = levels
        self.n_states = 2 ** (levels + 1) - 1

        super().__init__(params, levels, levels, self.n_states)

    def world(self):
        # each person has 2 children
        """
        e.g.
                   0
              1           2
           3    4      5    6
          7 8  9 10  11 12 13 14
        """
        # number of nodes = 2**0 + 2**1 + 2**3 + ... 2**levels = 2**(levels+1) - 1
        levels = self.levels
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

        self.adj, self.tran = adj, tran
        allowed_states = np.where(np.sum(self.adj, 1) > 0)[0]
        self.start_state = np.random.choice(allowed_states)

    def relation(self, s1, s2):
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

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):

        states_vec = np.zeros(self.n_states)
        choices = np.arange(self.par.s_size)

        for i in range(self.n_states):
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i] = new_state

        self.states_mat = states_vec.astype(int)

    def walk(self):
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))

        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1

        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)
            p = self.tran[int(position[i]), available]  # choose next position from actual allowed positions
            new_poss_pos = np.random.choice(available, p=p)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            rel_index, rel_type = self.relation(position[i], position[i + 1])

            direc[rel_index, i + 1] = 1

        return position, direc

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):

        raise ValueError('NEED TO ADD THIS FUNCTION FOR FAMILY TREE')


class LineTI(Environment):
    def __init__(self, params, length):
        self.length = length
        self.jump_length = params.env.jump_length
        self.n_states = length

        super().__init__(params, length, length, self.n_states)

    def world(self):
        length = self.length
        adj = np.zeros((length, length))

        for row in range(length):
            for col in range(length):
                diff = col - row
                if np.abs(diff) <= self.jump_length and diff != 0:
                    adj[row, col] = 1

        tran = np.zeros((length, length))
        for i in range(length):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran
        allowed_states = np.where(np.sum(self.adj, 1) > 0)[0]
        self.start_state = np.random.choice(allowed_states)

    def relation(self, s1, s2):
        diff = s2 - s1

        direc = np.sign(diff)
        num = np.abs(diff)

        direc_exact = diff + self.jump_length

        rel_name = str(direc) + ' ' + str(num)

        return direc, num, direc_exact, rel_name

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = np.arange(self.par.s_size)

        for i in range(self.n_states):
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i] = new_state

        self.states_mat = states_vec.astype(int)

    def walk(self):
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))

        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1
        direc[1, 0] = 1

        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)
            p = self.tran[int(position[i]), available]  # choose next position from actual allowed positions
            new_poss_pos = np.random.choice(available, p=p)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            d_, num_, _, _ = self.relation(position[i], position[i + 1])

            direc[0, i + 1] = d_
            direc[1, i + 1] = num_

        return position, direc

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        xs = [i for i in range(self.width)]
        ys = [0 for _ in xs]

        if cells is not None:
            cell_prepared = cp.deepcopy(cells).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


class Wood2000(Environment):
    def __init__(self, params, width, height):
        self.n_states = 4 * (height - 1 + width - 1) + 2 * (width - 1)
        super().__init__(params, width, height, self.n_states)

        # reward locations
        self.reward_pos = [height - 1 + width - 1, 3 * (height - 1 + width - 1)]
        # no reward locations
        self.no_reward_pos = [4 * (height - 1 + width - 1) + (width - 1) - 1,
                              4 * (height - 1 + width - 1) + 2 * (width - 1) - 1]
        self.reward_pos_training = self.reward_pos + self.no_reward_pos
        self.reward_value = self.n_states / 2

        self.start_state = 0

    def world(self):
        """
        26(NR) 25 24       27 28 29(NR)
        6(R)   5  4  3  15 16 17 18(R)
        7            2  14       19
        8            1  13       20
        9      10 11 0  12 23 22 21
        0,1,2,3 are same sense as 12,13,14,15
        """
        width = self.width
        height = self.height
        n_states_base = 2 * 2 * (width - 1 + height - 1)
        n_states = n_states_base + 2 * (width - 1)

        adj = np.zeros((n_states, n_states))

        for i in range(n_states_base - 1):
            adj[i, i + 1] = 1
        adj[n_states_base - 1, 0] = 1

        # ERRORS
        e_p = self.par.env.error_prob / (1 - self.par.env.error_prob)
        # 3 -> 27
        adj[height - 1, n_states_base + (width - 1)] = 1 * e_p
        # 15 -> 24
        adj[int(n_states_base / 2) + height - 1, n_states_base] = 1 * e_p
        # 26 -> 7
        adj[n_states_base + (width - 1) - 1, (height - 1 + width - 1) + 1] = 1
        # 29 -> 19
        adj[n_states_base + 2 * (width - 1) - 1, int(n_states_base / 2) + height - 1 + width] = 1

        for i in range(width - 2):
            adj[n_states_base + i, n_states_base + i + 1] = 1
            adj[n_states_base + (width - 1) + i, n_states_base + (width - 1) + i + 1] = 1

        tran = np.zeros((n_states, n_states))
        for i in range(n_states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        adj[adj > 0] = 1  # corrects adj matrix for error_prob introduced above!

        self.adj, self.tran = adj, tran

    def relation(self, s1, s2):
        """
        Use spatial actions.
        Equally could use a 'continue' action, and then a 'left/right' action for the choice points
        26(NR) 25 24       27 28 29(NR)
        6(R)   5  4  3  15 16 17 18(R)
        7            2  14       19
        8            1  13       20
        9      10 11 0  12 23 22 21
        """
        width = self.width
        height = self.height
        n_states_base = 2 * 2 * (width - 1 + height - 1)
        if s2 == 0 and s1 == n_states_base - 1:
            rel_type = 'left'
        elif s2 <= 1 * (height - 1) + 0 * (width - 1):
            rel_type = 'up'
        elif s2 <= 1 * (height - 1) + 1 * (width - 1):
            rel_type = 'left'
        elif s2 <= 2 * (height - 1) + 1 * (width - 1):
            rel_type = 'down'
        elif s2 <= 2 * (height - 1) + 2 * (width - 1):
            rel_type = 'right'
        elif s2 <= 3 * (height - 1) + 2 * (width - 1):
            rel_type = 'up'
        elif s2 <= 3 * (height - 1) + 3 * (width - 1):
            rel_type = 'right'
        elif s2 <= 4 * (height - 1) + 3 * (width - 1):
            rel_type = 'down'
        elif s2 < 4 * (height - 1) + 4 * (width - 1):  # note only less than from here!
            rel_type = 'left'
        # error bit left
        elif s2 < 4 * (height - 1) + 5 * (width - 1):
            rel_type = 'left'
        # error bit right
        elif s2 < 4 * (height - 1) + 6 * (width - 1):
            rel_type = 'right'
        else:
            raise ValueError('Wrong inputs given')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        width = self.width
        height = self.height
        states_vec = np.zeros(self.n_states)
        choices = np.arange(self.par.s_size)

        if self.par.use_reward:
            # choose reward sense
            reward_sense = np.random.choice(choices)
            no_reward_sense = np.random.choice(choices)
            # choices = np.delete(choices, reward_sense)
        else:
            reward_sense, no_reward_sense = 0, 0

        for i in range(self.n_states):
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i] = new_state

        for i in range(height):
            states_vec[i + 2 * (width - 1 + height - 1)] = states_vec[i]
        for i in range(width - 1):
            states_vec[4 * (width - 1 + height - 1) + i] = states_vec[height + i]
            states_vec[4 * (width - 1 + height - 1) + width - 1 + i] = states_vec[
                2 * (width - 1 + height - 1) + height + i]

        if self.par.use_reward:
            # make particular position special in track
            for r_p in self.reward_pos:
                states_vec[r_p] = reward_sense
            for r_p in self.no_reward_pos:
                states_vec[r_p] = no_reward_sense

        self.states_mat = states_vec.astype(int)

    def walk(self):
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))

        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1
        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)
            p = self.tran[int(position[i]), available]  # choose next position from actual allowed positions
            new_poss_pos = np.random.choice(available, p=p)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            # FOR THE ERROR TRIALS COULD BE A BIT MORE ANIMAL LIKE AND NOT REPEAT MISTAKE TWICE IN ROW ETC

            rel_index, rel_type = self.relation(position[i], position[i + 1])

            direc[rel_index, i + 1] = 1

        return position, direc

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        width = self.width
        height = self.height
        xs = []
        ys = []
        cell_prepared = []
        cells_ = None
        if cells is not None:
            cells_ = cp.deepcopy(cells).flatten().tolist()

        structure_x = [0 for _ in range(height - 1)] + [- i for i in range(width - 1)] + \
                      [-(width - 1) for _ in range(height - 1)] + [-(width - 1) + i for i in range(width - 1)]
        structure_y = [i for i in range(height - 1)] + [height - 1 for _ in range(width - 1)] + \
                      [height - 1 - i for i in range(height - 1)] + [0 for _ in range(width - 1)]

        shifts = [0,
                  2 * (height - 1 + width - 1),
                  4 * (height - 1 + width - 1),
                  4 * (height - 1 + width - 1) + width - 1]
        mults = [1, -1, 1, -1]

        for route in range(4 if _plot_specs.wood2000.plot_all else 2):
            min_x = np.min([mults[route] * x for x in structure_x])
            for i in range(2 * (height - 1 + width - 1) if _plot_specs.wood2000.plot_all else height + width - 1):
                xs.append(mults[route] * structure_x[i] - min_x + route * (
                        width + (2 if _plot_specs.wood2000.plot_all else 1)))
                ys.append(structure_y[i])
                if cells is not None:
                    if route == 2:  # error from 12 -> 24
                        if i < height:
                            cell_prepared.append(cells_[i + shifts[1]])
                        elif i > (height - 1 + width - 1):
                            cell_prepared.append(cells_[i + shifts[0]])
                        else:
                            cell_prepared.append(cells_[i - height + shifts[route]])
                    elif route == 3:  # 3 -> 27
                        if i < height:
                            cell_prepared.append(cells_[i + shifts[0]])
                        elif i > (height - 1 + width - 1):
                            cell_prepared.append(cells_[i + shifts[1]])
                        else:
                            cell_prepared.append(cells_[i - height + shifts[route]])
                    else:
                        cell_prepared.append(cells_[i + shifts[route]])

        if cells is not None:
            cell_prepared = np.asarray(cell_prepared).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


class Frank2000(Environment):
    def __init__(self, params, width, height):
        self.n_states = 6 * height + 6 * width + 4 * (height - 1)
        super().__init__(params, width, height, self.n_states)

        # reward locations
        self.reward_pos = [0, 2 * height + width - 1, 3 * height + 2 * width + 1 * (height - 1) - 1,
                           4 * height + 3 * width + 2 * (height - 1) - 1]
        # no reward locations
        self.no_reward_pos = [5 * height + 5 * width + 4 * (height - 1) - 1,
                              6 * height + 6 * width + 4 * (height - 1) - 1]
        self.reward_pos_training = self.reward_pos + self.no_reward_pos
        self.reward_value = self.n_states / 2

        self.start_state = 0

    def world(self):
        """
        47(NR) ->10  9(R)        0(R)  18(R)        27(R)   41(NR) -> 28
        46           10 8        1  17 35 19       26 28    40
        45           11 7        2  16 34 20       25 29    39
                        6  5  4  3        21 22 23 24
                     12    13 14    15 33    32 31    30
                        44 43 42 <-21    3-> 36 37 38
                                or 33   or 15
        (wrong way up)
        6,7,8 are same sense as 12,11,10 etc
        """
        width = self.width
        height = self.height
        n_states_base = 4 * (height - 1 + height + width)
        n_states = n_states_base + 2 * (width + height)

        # n_states_base = 12 * (width - 1)
        # n_states = n_states_base + 4 * (width - 1)

        adj = np.zeros((n_states, n_states))

        for i in range(n_states_base - 1):
            adj[i, i + 1] = 1
        adj[n_states_base - 1, 0] = 1

        e_p = self.par.env.error_prob / (1 - self.par.env.error_prob)

        # ERRORS
        # 3 -> 36
        adj[height - 1, n_states_base] = 1 * e_p
        # 41 -> 28
        adj[n_states_base + (width - 1) + height, 4 * height + 3 * width + 2 * (height - 1)] = 1
        # 21 -> 42
        adj[3 * height + 2 * width + 2 * (height - 1) - 1, n_states_base + width + height] = 1 * e_p
        # 47 -> 10
        adj[n_states_base + 2 * (width + height) - 1, 2 * height + width] = 1
        # 33 -> 42
        adj[n_states_base - (height - 1), n_states_base + (width + height)] = 1 * e_p
        # 15 -> 36
        adj[2 * (height + width) + height - 1, n_states_base] = 1 * e_p

        for i in range(width + height - 1):
            adj[n_states_base + i, n_states_base + i + 1] = 1
            adj[n_states_base + width + height + i, n_states_base + width + height + i + 1] = 1

        tran = np.zeros((n_states, n_states))
        for i in range(n_states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        adj[adj > 0] = 1  # corrects adj matrix for error_prob introduced above!

        self.adj, self.tran = adj, tran

    def relation(self, _s1, s2):
        """
        Use spatial actions
        Equally could use a 'continue' action, and then a 'left/right' action for the choice points
        47(NR) ->10  9(R)        0(R)  18(R)        27(R)   41(NR) -> 28
        46           10 8        1  17 35 19       26 28    40
        45           11 7        2  16 34 20       25 29    39
                        6  5  4  3        21 22 23 24
                     12    13 14    15 33    32 31    30
                        44 43 42 <-21    3-> 36 37 38
                                or 33   or 15
        """
        width = self.width
        height = self.height
        if s2 == 0:
            rel_type = 'up'
        elif s2 < 1 * height + 0 * width + 0 * (height - 1):
            rel_type = 'down'
        elif s2 < 1 * height + 1 * width + 0 * (height - 1) + 1:
            rel_type = 'left'
        elif s2 < 2 * height + 1 * width + 0 * (height - 1):
            rel_type = 'up'
        elif s2 < 2 * height + 1 * width + 1 * (height - 1):
            rel_type = 'down'
        elif s2 < 2 * height + 2 * width + 1 * (height - 1) + 1:
            rel_type = 'right'
        elif s2 < 3 * height + 2 * width + 1 * (height - 1):
            rel_type = 'up'
        elif s2 < 3 * height + 2 * width + 2 * (height - 1):
            rel_type = 'down'
        elif s2 < 3 * height + 3 * width + 2 * (height - 1) + 1:
            rel_type = 'right'
        elif s2 < 4 * height + 3 * width + 2 * (height - 1):
            rel_type = 'up'
        elif s2 < 4 * height + 3 * width + 3 * (height - 1):
            rel_type = 'down'
        elif s2 < 4 * height + 4 * width + 3 * (height - 1) + 1:
            rel_type = 'left'
        elif s2 < 4 * height + 4 * width + 4 * (height - 1):
            rel_type = 'up'
        # error right bit
        elif s2 < 4 * height + 5 * width + 4 * (height - 1) + 1:
            rel_type = 'right'
        elif s2 < 5 * height + 5 * width + 4 * (height - 1):
            rel_type = 'up'
        # error left bit
        elif s2 < 5 * height + 6 * width + 4 * (height - 1) + 1:
            rel_type = 'left'
        elif s2 < 6 * height + 6 * width + 4 * (height - 1):
            rel_type = 'up'
        else:
            raise ValueError('Wrong inputs given')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = np.arange(self.par.s_size)
        width = self.width
        height = self.height

        if self.par.use_reward:
            # choose reward sense
            reward_sense = np.random.choice(choices)
            no_reward_sense = np.random.choice(choices)
            # choices = np.delete(choices, reward_sense)
        else:
            reward_sense, no_reward_sense = 0, 0

        for i in range(self.n_states):
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i] = new_state

        """
        Use spatial actions
        Equally could use a 'continue' action, and then a 'left/right' action for the choice points
        47(NR) ->10  9(R)        0(R)  18(R)        27(R)   41(NR) -> 28
        46           10 8        1  17 35 19       26 28    40
        45           11 7        2  16 34 20       25 29    39
                        6  5  4  3        21 22 23 24
                     12    13 14    15 33    32 31    30
                        44 43 42 <-21    3-> 36 37 38
                                or 33   or 15
        """
        repeat_len = height + width + height - 1
        for i in range(height):
            # make center bits the same
            states_vec[2 * repeat_len + i] = states_vec[i]
        for i in range(repeat_len):
            # make forward and back on left section same
            states_vec[2 * repeat_len - i] = states_vec[i]
        for i in range(repeat_len - 1):
            # make forward and back on right section same
            states_vec[4 * repeat_len - i - 1] = states_vec[2 * repeat_len + i + 1]
        # MAKE ERROR BITS TOO
        n_states_base = 4 * repeat_len
        for i in range(width + height):
            states_vec[n_states_base + i] = states_vec[2 * repeat_len + height + i]
            states_vec[n_states_base + width + height + i] = states_vec[height + i]

        if self.par.use_reward:
            # make particular position special in track
            for r_p in self.reward_pos:
                states_vec[r_p] = reward_sense
            for r_p in self.no_reward_pos:
                states_vec[r_p] = no_reward_sense

        self.states_mat = states_vec.astype(int)

    def walk(self):
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))

        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1

        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)
            p = self.tran[int(position[i]), available]  # choose next position from actual allowed positions
            new_poss_pos = np.random.choice(available, p=p)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            rel_index, rel_type = self.relation(position[i], position[i + 1])

            direc[rel_index, i + 1] = 1

        return position, direc

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        width = self.width
        height = self.height
        xs = []
        ys = []
        cell_prepared = []
        cells_ = None
        if cells is not None:
            cells_ = cp.deepcopy(cells).flatten().tolist()

        # plot 4 routes + 2 error paths?
        structure_x = [0 for _ in range(height)] + [i + 1 for i in range(width)] + [width + 1 for _ in
                                                                                    range(height)]
        structure_y = [i for i in range(height)] + [height - 1 for _ in range(width)] + [height - 1 - i for i in
                                                                                         range(height)]
        order = [1, 0, 2, 3, 4, 5]
        shifts = [0,
                  2 * height + 1 * width + 0 * (height - 1) - 1,
                  3 * height + 2 * width + 1 * (height - 1) - 1,
                  4 * height + 3 * width + 2 * (height - 1) - 1,
                  4 * height + 4 * width + 3 * (height - 1) - 1,
                  5 * height + 5 * width + 3 * (height - 1) - 1]
        mults = [-1, -1, 1, 1, 1, -1]
        for shift, route in enumerate(order[:6 if _plot_specs.frank2000.plot_all else 4]):
            structure_x_use = structure_x[::-1] if route in [1, 3] else structure_x
            min_x = np.min([mults[route] * x for x in structure_x])
            for i in range(2 * height + width):
                xs.append(mults[route] * structure_x_use[i] - min_x + shift * (width + 3))
                ys.append(structure_y[i])
                if cells is not None:
                    if route == 3 and i == 2 * height + width - 1:
                        cell_prepared.append(cells_[0])
                    elif route == 4 and i < height:
                        cell_prepared.append(cells_[i])
                    elif route == 5 and i < height:
                        cell_prepared.append(cells_[i + 3 * height + 2 * width + 1 * (height - 1) - 1])
                    else:
                        cell_prepared.append(cells_[i + shifts[route]])

        if cells is not None:
            cell_prepared = np.asarray(cell_prepared).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


class Grieves2016(Environment):

    def __init__(self, params, seg_len):
        self.seg_len = seg_len
        self.map_states = 7 * seg_len + 8 + 4 + 1
        self.n_states = 8 * self.map_states
        super().__init__(params, seg_len, seg_len, self.n_states)

        # choices: middle, left, right
        self.choice_points = [1 * seg_len + 1, 2 * seg_len + 2, 5 * seg_len + 7]
        # goal boxes: left, center-left, center-right, box right
        self.goal_boxes = [3 * seg_len + 3 + 0, 4 * seg_len + 4 + 1, 6 * seg_len + 6 + 2, 7 * seg_len + 7 + 3]
        # reward locations
        self.reward_pos = sorted(
            [(a + 1) * self.map_states - 1 for a in [0, 1, 2, 3]] + [b * self.map_states + a + 1 for a, b in
                                                                     zip(self.goal_boxes, [4, 5, 6, 7])])
        # no reward locations
        self.no_reward_pos = sorted(
            [b * self.map_states + a + 1 for a, b in itertools.product(self.goal_boxes, [0, 1, 2, 3, 4, 5, 6, 7]) if
             b * self.map_states + a + 1 not in self.reward_pos])
        # reward points for training
        self.reward_pos_training = self.reward_pos[:4] + [x for x in self.no_reward_pos if
                                                          x < 4 * self.map_states and x % self.map_states !=
                                                          self.goal_boxes[
                                                              np.floor(x / self.map_states).astype(int)] + 1]
        self.reward_value = self.n_states / 4

        try:
            self.par.env.simplified
        except (AttributeError, KeyError):
            self.par.env.simplified = False

        if self.par.env.simplified:
            allowed_states = [0]  # always state in map 0
        else:
            allowed_states = [self.map_states * a for a in [4, 5, 6, 7]]  # i.e. at beginning of 'exploration' maps
        self.start_state = np.random.choice(allowed_states)

    def world(self):
        """
        We are modelling this without the see-through barrier, as it complicates things
        Instead, we only give rewards in the center goal box if the animal took the correct route
        e.g. with segment length 1
        # maps where the rodent knows where the goal is
              10-NR  15-NR         30-NR 39-R  35-NR          50-NR  55-NR 59-R        70-NR  75-NR
                  9   14                  29  34                  49  54                  69  74
                  8   13                  28  33                  48  53                  68  73
                4       12              24      32              44      52              64      72
              5   3   11  16          25  23  31  36          45  43  51  56          65  63  71  76
            6       2       17      26      22      37      46      42      57      66      62      77
        7-NR 19-R   1     18-NR   27-NR     21    38-NR   47-NR     41    58-NR   67-NR     61     78-NR 79-R
                    0                       20                      40                      60
        [9,14], [29,34], [49,54], [69,74] same sense (have separated these so there is a spatial element!)
        Different maps have identical sense (apart from reward state)
        Predicting reward states are necessary to make reps not just spatial
        Clearly reward action not ideal, but keeping it in here until we can make inferences based on
        unexpected sensory data.
        19 -> 0 (R act)           39 -> 20 (R act)           59 -> 40 (R act)          79 -> 60 (R act)
        10,15,18 -> 0 (NR act)    27,35,38 -> 20 (NR act)    47,50,58 -> 60 (NR act)   67,70,75 -> 60 (NR act)
        7 -> 100,120,140 (NR act)  30 -> 80,120,140 (NR act)  55 -> 80,100,140 (NR act)  78 -> 80,100,120 (NR act)

        # maps where the rodent is exploring after discovering a NR at a previously rewarded route
        # start state should be randomly chosen from 80, 100, 120, 140
        # have added in an extra 'alone' state so all maps have the same size
               90-NR  95-NR             110-R 115-NR           130-NR 135-R             150-NR 155-NR
                  89  94                  109 114                 129 134                  149 154
                  88  93     99(alone)    108 113    119(alone)   128 133     139(alone)   148 153     159(alone)
                84      92              104     112             124    132               144     152
              85  83  91  96          105 103 111 116         125 123 131 136          145 143 151  156
            86      82      97      106     102     117     126     122     137      146     142      157
           87-R     81    98-NR   107-NR    101   118-NR  127-NR    121   138-NR   147-NR    141     158-NR
                    80                      100                     120                      140
        87 -> 0 (R)                 110 -> 20 (R)           135 -> 40 (R)           158 -> 60 (R)
        90,95,98 -> 80 (NR)         107,115,118 -> 100 (NR)  127,130,138 -> 120 (NR)  147,110,155 -> 140 (NR)
        # segments = 7 , # 'nodes' = 8 , # reward = 4
        """
        map_s = self.map_states
        n_states = 8 * map_s
        adj = np.zeros((n_states, n_states))
        # add transitions to reward states (scale so that maps switch i.e. in map 0 more likely to go to R than NR)
        s_p = (1 - self.par.env.switch_prob) / self.par.env.switch_prob
        # add errors, i.e. taking wrong choice at choice points
        e_p = self.par.env.error_prob / (1 - self.par.env.error_prob)
        # choices: middle, left, right
        choice_points = self.choice_points
        # goal boxes: left, center-left, center-right, box right
        goal_boxes = self.goal_boxes

        # sort out basic 'forwards' transitions
        for map_i in range(8):
            # shift to start of each map
            sh_m = map_i * map_s
            # beginning to left no-reward
            for i in range(3 * self.seg_len + 4):
                adj[sh_m + i, sh_m + i + 1] = 1
            # left choice to left-center no-reward
            adj[sh_m + choice_points[1], sh_m + choice_points[1] + self.seg_len + 3] = 1
            for i in range(self.seg_len + 2 - 1):
                adj[sh_m + choice_points[1] + self.seg_len + 3 + i, sh_m + choice_points[
                    1] + self.seg_len + 3 + i + 1] = 1
            # middle choice to centre right no-reward
            adj[sh_m + choice_points[0], sh_m + choice_points[0] + 3 * self.seg_len + 6] = 1
            for i in range(2 * self.seg_len - 1 + 3):
                adj[sh_m + choice_points[0] + 3 * self.seg_len + 6 + i, sh_m + choice_points[
                    0] + 3 * self.seg_len + 6 + i + 1] = 1
            # right choice to right no-reward
            adj[sh_m + choice_points[2], sh_m + choice_points[2] + self.seg_len + 3] = 1
            for i in range(self.seg_len + 2 - 1):
                adj[sh_m + choice_points[2] + self.seg_len + 3 + i, sh_m + choice_points[
                    2] + self.seg_len + 3 + i + 1] = 1

            # include reverse actions for exploration maps
            if map_i >= 4:
                # (note: in walk function there is a bias towards goal un-explored goal states)
                adj[map_i * map_s:(map_i + 1) * map_s, map_i * map_s:(map_i + 1) * map_s] += \
                    adj[map_i * map_s:(map_i + 1) * map_s, map_i * map_s:(map_i + 1) * map_s].T
                # remove states from goal boxes backwards and reward-states backwards
                for goal in goal_boxes:
                    adj[sh_m + goal, sh_m + goal - 1] = 0
                    adj[sh_m + goal + 1, sh_m + goal] = 0

        c_ps = [1, 1, 2, 2]  # note choice point 0 included in loop below...
        error_points_add_1 = [3 * self.seg_len + 6, 3 * self.seg_len + 6, 1, 1]
        error_points_add_2 = [self.seg_len + 3, 1, self.seg_len + 3, 1]
        for map_i, (c_p, error_p_1, error_p_2) in enumerate(zip(c_ps, error_points_add_1, error_points_add_2)):
            # add switches i.e. in map 0 go to NR not R
            adj[map_i * map_s + goal_boxes[map_i], (map_i + 1) * map_s - 1] = 1 * s_p
            # add errors, i.e. taking wrong choice at choice points
            adj[map_i * map_s + choice_points[0], map_i * map_s + choice_points[0] + error_p_1] = 1 * e_p
            adj[map_i * map_s + choice_points[c_p], map_i * map_s + choice_points[c_p] + error_p_2] = 1 * e_p

            # connect map ends to map starts
            # reward back to start
            adj[(map_i + 1) * map_s - 1, map_i * map_s] = 1

            # no-reward switch
            if self.par.env.simplified:
                # miss out exploration phase after reward moves - instead receive action for each specific sub-map
                adj[map_i * map_s + goal_boxes[map_i] + 1, [x * map_s for x in
                                                            [x for x in [0, 1, 2, 3] if x != map_i]]] = 1
            else:
                # go to exploration map
                adj[map_i * map_s + goal_boxes[map_i] + 1, [x * map_s for x in
                                                            [x for x in [4, 5, 6, 7] if x != map_i + 4]]] = 1
            # error no-reward stay in map
            adj[[map_i * map_s + x + 1 for x in
                 [x for x in goal_boxes if x != goal_boxes[map_i]]], map_i * map_s] = 1

            # exploration maps
            # reward to map 0
            adj[(map_i + 4) * map_s + goal_boxes[map_i] + 1, map_i * map_s] = 1
            # no reward explore
            adj[[(map_i + 4) * map_s + x + 1 for x in [x for x in goal_boxes if x != goal_boxes[map_i]]],
                (map_i + 4) * map_s] = 1

        tran = np.zeros((n_states, n_states))
        for i in range(n_states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        adj[adj > 0] = 1  # corrects adj matrix for error_prob introduced above!

        self.adj, self.tran = adj, tran

    def relation(self, s1, s2):
        """
              10-NR  15-NR         30-NR 39-R  35-NR          50-NR  55-NR 59-R        70-NR  75-NR
                  9   14                  29  34                  49  54                  69  74
                  8   13                  28  33                  48  53                  68  73
                4       12              24      32              44      52              64      72
              5   3   11  16          25  23  31  36          45  43  51  56          65  63  71  76
            6       2       17      26      22      37      46      42      57      66      62      77
        7-NR 19-R   1     18-NR   27-NR     21    38-NR   47-NR     41    58-NR   67-NR     61     78-NR 79-R
                    0                       20                      40                      60

               90-NR  95-NR             110-R 115-NR           130-NR 135-R             150-NR 155-NR
                  89  94                  109 114                 129 134                  149 154
                  88  93     99(alone)    108 113    119(alone)   128 133     139(alone)   148 153     159(alone)
                84      92              104     112             124    132               144     152
              85  83  91  96          105 103 111 116         125 123 131 136          145 143 151  156
            86      82      97      106     102     117     126     122     137      146     142      157
           87-R     81    98-NR   107-NR    101   118-NR  127-NR    121   138-NR   147-NR    141     158-NR
                    80                      100                     120                      140
        """
        seg_len = self.seg_len
        map_states = self.map_states
        reward_locations = self.reward_pos
        no_reward_locations = self.no_reward_pos
        choice_points = self.choice_points
        goal_boxes = self.goal_boxes
        map_loc_from = s1 % map_states
        map_loc_to = s2 % map_states

        if map_loc_to == 0 and s1 in reward_locations:
            # 19 -> 0 ... 39 -> 20 ... 59 -> 40 ... 79 -> 60  (also exploration to intentional phase e.g. 87 -> 0)
            rel_type = 'reward'
        elif self.par.env.simplified and (
                map_loc_to == 0 and s1 in [map_states * map_i + goal_boxes[map_i] + 1 for map_i in [0, 1, 2, 3]]):
            # this is for simplified version i.e 7 -> 20,40,60 or 55 -> 0,20,60
            rel_type = str(int(s2 / map_states))
        elif map_loc_to == 0 and s1 in no_reward_locations:
            # e.g. for map 1: [10, 15, 18] -> 0
            rel_type = 'no-reward'
        elif s2 in reward_locations + no_reward_locations and map_loc_from in goal_boxes:
            # e.g. for map 1: 6 -> 7/19 ... 9 -> 10 ... 14 -> 15 ... 17 -> 18
            rel_type = 'try reward'
        elif map_loc_from < map_loc_to <= choice_points[0]:
            # 0 -> 2
            rel_type = 'up'
        elif map_loc_to < map_loc_from and map_loc_to < choice_points[0]:
            # 2 -> 0
            rel_type = 'down'
        elif (map_loc_to > map_loc_from and choice_points[1] - seg_len <= map_loc_to <= choice_points[1]) or \
                (map_loc_to > map_loc_from and goal_boxes[2] - seg_len <= map_loc_to <= goal_boxes[2]) or \
                (map_loc_to < map_loc_from and goal_boxes[3] - seg_len <= map_loc_from <= goal_boxes[3]):
            # 2 -> 4 ... 12 -> 14 ... 17 -> 12
            rel_type = 'up-left'
        elif (map_loc_to < map_loc_from and choice_points[1] - seg_len <= map_loc_from <= choice_points[1]) or \
                (map_loc_to < map_loc_from and goal_boxes[2] - seg_len <= map_loc_from <= goal_boxes[2]) or \
                (map_loc_to > map_loc_from and goal_boxes[3] - seg_len <= map_loc_to <= goal_boxes[3]):
            # 4 -> 2 ... 14 -> 12 ... 12 -> 17
            rel_type = 'down-right'
        elif (map_loc_to < map_loc_from and choice_points[2] - seg_len <= map_loc_from <= choice_points[2]) or \
                (map_loc_to > map_loc_from and goal_boxes[0] - seg_len <= map_loc_to <= goal_boxes[0]) or \
                (map_loc_to < map_loc_from and goal_boxes[1] - seg_len <= map_loc_from <= goal_boxes[1]):
            # 12 -> 2 ... 4 -> 6 ... 9 -> 4
            rel_type = 'down-left'
        elif (map_loc_to > map_loc_from and choice_points[2] - seg_len <= map_loc_to <= choice_points[2]) or \
                (map_loc_to < map_loc_from and goal_boxes[0] - seg_len <= map_loc_from <= goal_boxes[0]) or \
                (map_loc_to > map_loc_from and goal_boxes[1] - seg_len <= map_loc_to <= goal_boxes[1]):
            # 2 -> 12 ... 6 -> 4 ... 4 -> 9
            rel_type = 'up-right'
        else:
            print(s1, s2, map_loc_from, map_loc_to)
            raise ValueError('Impossible transition')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = np.arange(self.par.s_size)

        if self.par.use_reward:
            # choose reward sense
            reward_sense = np.random.choice(choices)
            no_reward_sense = np.random.choice(choices)
            # choices = np.delete(choices, reward_sense)
        else:
            reward_sense, no_reward_sense = 0, 0

        for i in range(self.n_states):
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i] = new_state
        # make center state the same sense
        states_vec[6 * self.seg_len + 8] = states_vec[4 * self.seg_len + 5]
        # make all other maps same sense
        map_states = self.map_states
        for map_i in range(1, 8):
            map_sh = map_i * map_states
            first_states = [map_sh + ss for ss in range(map_states) if ss not in self.reward_pos + self.no_reward_pos]
            second_states = [ss - map_states for ss in first_states]
            states_vec[first_states] = states_vec[second_states]

        if self.par.use_reward:
            # make particular position special in track
            for r_p in self.reward_pos:
                states_vec[r_p] = reward_sense
            for r_p in self.no_reward_pos:
                states_vec[r_p] = no_reward_sense

        self.states_mat = states_vec.astype(int)

    def walk(self):
        time_steps = self.walk_len

        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))

        goal_boxes = self.goal_boxes
        map_states = self.map_states

        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1

        maps = [x for x in range(4)]
        map_i = np.floor(position[0] / map_states).astype(int)
        visited = []

        # work out num of steps from each state to goal state
        if map_i >= 4:
            # note adj_ij is connection from i to j
            partition_start, partition_stop = map_i * map_states, (map_i + 1) * map_states
            adj_exploration = self.adj[partition_start:partition_stop, partition_start:partition_stop]
            steps_to_mat = np.zeros_like(adj_exploration)
            adj_exploration_n_steps = np.eye(map_states)
            for i in range(map_states):
                adj_exploration_n_steps = np.matmul(adj_exploration, adj_exploration_n_steps)
                steps_to_mat[np.logical_and(adj_exploration_n_steps > 0, steps_to_mat == 0)] = i + 1
            # set diagonal to zero
            np.fill_diagonal(steps_to_mat, 0)
            distances_to_goals = steps_to_mat[:, goal_boxes]
        else:
            distances_to_goals = None

        goal_guess = 0
        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)

            # If on exploration phase - choose goal location from possibilities and bias actions towards them.
            # If incorrect don't choose same one next time.
            # If just come from a map then don choose that one either.
            map_i = np.floor(position[i] / map_states).astype(int)
            map_location = position[i] % map_states
            if map_i < 4:
                # clear list of visited maps
                visited = []
                # transitions randomly
                p = self.tran[int(position[i]), available]  # choose next position from actual allowed positions
                new_poss_pos = np.random.choice(available, p=p)
            else:
                if map_location == 0:
                    # Bias towards unvisited goal states. Only do once at beginning of each exploration
                    goal_guess = np.random.choice(list(set(maps).symmetric_difference(set(visited))))
                    # print(current_map, maps, visited, goal_guess)
                distances = distances_to_goals[:, goal_guess][available % map_states]
                closest_state = available[np.argmin(distances)]

                r_n = np.random.rand()
                if r_n > self.par.env.exploration_bias or map_location in goal_boxes + [x + 1 for x in goal_boxes]:
                    # transition randomly certain proportion of time or if at goal / reward / no-reward state
                    p = self.tran[int(position[i]), available]  # choose next position from actual allowed positions
                    new_poss_pos = np.random.choice(available, p=p)
                else:
                    # transition biased
                    new_poss_pos = closest_state

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                raise ValueError("Impossible transition: state " + str(position[i]) + ' to state ' + str(new_poss_pos))

            # If just visited goal state (s in goal_boxes), add that map to visited
            if map_location in [x + 1 for x in goal_boxes]:
                # if current_map >= 4:
                visited.append(goal_boxes.index(map_location - 1))
                visited = list(set(visited))

            rel_index, rel_type = self.relation(position[i], position[i + 1])

            direc[rel_index, i + 1] = 1

        return position, direc

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        width = self.seg_len
        xs = []
        ys = []
        cell_prepared = []
        cells_ = None
        if cells is not None:
            cells_ = cp.deepcopy(cells).flatten().tolist()

        """
              10-NR  15-NR        39-R 30-NR  35-NR            50-NR  55-NR 59-R       70-NR  75-NR
                  9   14                  29  34                  49  54                  69  74
                  8   13                  28  33                  48  53                  68  73
                4       12              24      32              44      52              64      72
              5   3   11  16          25  23  31  36          45  43  51  56          65  63  71  76
            6       2       17      26      22      37      46      42      57      66      62      77
        7-NR 19-R   1     18-NR   27-NR     21    38-NR   47-NR     41    58-NR   67-NR     61  79-R  78-NR 
                    0                       20                      40                      60
        We assume the first arm splits at 90 degrees
        Let's ignore the reward bits.
        We really should make reward a separate thing to sensory stuff. But still have a memory for it etc.
        Then can predict both sensory and reward at each step.
        """
        scale_x = 1 / np.sqrt(2)  # np.sqrt(3) / 2
        scale_y = 1 / np.sqrt(2)  # 1 / 2
        x_s_c0 = [0 for _ in range(width + 1)]
        x_c0_c1 = [- i * scale_x for i in range(width + 1)]
        x_c1_g0 = [-(width + 1 + i) * scale_x for i in range(width + 1)]
        x_c1_g1 = [-(width + 1 - i) * scale_x for i in range(width + 1)]
        x_c0_c2 = [i * scale_x for i in range(width + 1)]
        x_c2_g2 = [(width + 1 - i) * scale_x for i in range(width + 1)]
        x_c2_g3 = [(width + 1 + i) * scale_x for i in range(width + 1)]

        y_s_c0 = [i for i in range(width + 1)]
        y_c0_c1 = [width + 1 + i * scale_y for i in range(width + 1)]
        y_c1_g0 = [width + 1 + (width + 1 - i) * scale_y for i in range(width + 1)]
        y_c1_g1 = [width + 1 + (width + 1 + i) * scale_y for i in range(width + 1)]
        y_c0_c2 = [width + 1 + i * scale_y for i in range(width + 1)]
        y_c2_g2 = [width + 1 + (width + 1 + i) * scale_y for i in range(width + 1)]
        y_c2_g3 = [width + 1 + (width + 1 - i) * scale_y for i in range(width + 1)]

        central_shift = 2 * scale_x if _plot_specs.grieves2016.plot_all else 0
        x_g0 = [-2 * (width + 1) * scale_x, (-2 * (width + 1) - 1) * scale_x]
        x_g1 = [-central_shift, -central_shift - scale_x]
        x_g2 = [central_shift, central_shift - scale_x]
        x_g3 = [2 * (width + 1) * scale_x, (2 * (width + 1) - 1) * scale_x]

        y_g0 = [width + 1, width + 1 - scale_y]
        y_g1 = [width + 1 + 2 * (width + 1) * scale_y, width + 1 + 2 * (width + 1) * scale_y + scale_y]
        y_g2 = [width + 1 + 2 * (width + 1) * scale_y, width + 1 + 2 * (width + 1) * scale_y + scale_y]
        y_g3 = [width + 1, width + 1 - scale_y]

        x_rewards = [x + scale_x for x in [x_g0[0], x_g1[0], x_g2[0], x_g3[0]]]
        x_rewards = x_rewards + x_rewards
        y_rewards = [y for y in [y_g0[-1], y_g1[-1], y_g2[-1], y_g3[-1]]]
        y_rewards = y_rewards + y_rewards

        x_structure = [x_s_c0, x_c0_c1, x_c1_g0, x_g0, x_c1_g1, x_g1, x_c0_c2, x_c2_g2, x_g2, x_c2_g3, x_g3]
        y_structure = [y_s_c0, y_c0_c1, y_c1_g0, y_g0, y_c1_g1, y_g1, y_c0_c2, y_c2_g2, y_g2, y_c2_g3, y_g3]

        skip = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]
        map_shift = 11 if _plot_specs.grieves2016.plot_all else 2
        state_index = 0
        routes = [[0, 1, 2, 3], [0, 1, 4, 5], [0, 6, 7, 8], [0, 6, 9, 10]]
        map_choice_hack = [[self.choice_points[1]], [self.choice_points[1]], [self.choice_points[0]],
                           [self.choice_points[0], self.choice_points[2]]]
        map_choice_hack = [[x + i * self.map_states for x in ccc_ppp] for i, ccc_ppp in enumerate(map_choice_hack)]
        for map_i in range(8 if _plot_specs.grieves2016.plot_all else 4):
            for section, (xs_, ys_, sk) in enumerate(zip(x_structure, y_structure, skip)):
                if not _plot_specs.grieves2016.plot_all:
                    sk = 0
                for seq_num, (x, y) in enumerate(zip(xs_[sk:], ys_[sk:])):
                    if _plot_specs.grieves2016.plot_all or section in routes[map_i][:-1] or (
                            section == routes[map_i][-1] and seq_num == 0):
                        xs.append(x + map_i * map_shift)
                        ys.append(y)
                        if cells is not None:
                            if seq_num == 0 and sk != skip[section]:
                                hack_popped = map_choice_hack[map_i].pop(0)
                                # print(state_index, hack_popped, 'hack')
                                cell_prepared.append(cells_[hack_popped])
                            else:
                                # print(state_index, state_index)
                                cell_prepared.append(cells_[state_index])
                    if seq_num == 0 and sk != skip[section]:
                        state_index += 0
                    else:
                        state_index += 1
            if _plot_specs.grieves2016.plot_all and state_index not in [self.map_states * x - 1 for x in [5, 6, 7, 8]]:
                # plot reward positions
                # print(state_index, 'reward')
                xs.append(x_rewards[map_i] + map_i * map_shift)
                ys.append(y_rewards[map_i])
                if cells is not None:
                    cell_prepared.append(cells_[state_index])
            state_index += 1

        if cells is not None:
            cell_prepared = np.asarray(cell_prepared).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


class Sun2020(Environment):

    def __init__(self, params, width):
        self.n_laps = params.sun2020.n_laps
        self.n_states = self.n_laps * (2 * width + 2 * (width - 2))
        super().__init__(params, width, width, self.n_states)

        # reward locations
        self.reward_pos = [self.n_states - 1]
        self.no_reward_pos = []
        self.reward_pos_training = self.reward_pos + self.no_reward_pos
        self.reward_value = self.n_states

        self.start_state = 0

    def world(self):
        width = self.width
        n_states = self.n_laps * (2 * width + 2 * (width - 2))

        adj = np.zeros((n_states, n_states))

        # go round track twice
        for i in range(n_states):
            if i < n_states - 1:
                adj[i, i + 1] = 1

        # lap to beginning:
        adj[n_states - 1, 0] = 1

        tran = np.zeros((n_states, n_states))
        for i in range(n_states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran

    def relation(self, s1, s2):
        width = self.width
        n_states = self.n_laps * (2 * width + 2 * (width - 2))
        pos_lap_1 = s1 % int(n_states / self.n_laps)
        pos_lap_2 = s2 % int(n_states / self.n_laps)

        if s1 > n_states or s2 > n_states:
            raise ValueError('impossible state index - too high')
        if pos_lap_2 - pos_lap_1 == 0:
            rel_type = 'stay still'
        elif s2 - s1 != 1 and s2 - s1 != -(n_states - 1):
            raise ValueError('impossible state transition')
        elif 0 < pos_lap_2 < width:
            rel_type = 'right'
        elif width <= pos_lap_2 < width + 1 * (width - 1):
            rel_type = 'up'
        elif width + 1 * (width - 1) <= pos_lap_2 < width + 2 * (width - 1):
            rel_type = 'left'
        elif pos_lap_2 < width + 3 * (width - 1):
            rel_type = 'down'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):

        states_vec = np.zeros(self.n_states)
        choices = np.arange(self.par.s_size)

        if self.par.use_reward:
            # choose reward sense
            reward_sense = np.random.choice(choices)
            no_reward_sense = np.random.choice(choices)
            # choices = np.delete(choices, reward_sense)
        else:
            reward_sense, no_reward_sense = 0, 0

        for i in range(self.n_states):
            new_state = np.random.choice(choices)
            len_loop = int(self.n_states / self.n_laps)
            states_vec[i] = new_state if i / len_loop < 1 else states_vec[i - len_loop]

        if self.par.use_reward:
            # make particular position special in track
            for r_p in self.reward_pos:
                states_vec[r_p] = reward_sense
            for r_p in self.no_reward_pos:
                states_vec[r_p] = no_reward_sense

        self.states_mat = states_vec.astype(int)

    def walk(self):
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))

        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1
        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)
            p = self.tran[int(position[i]), available]  # choose next position from actual allowed positions
            new_poss_pos = np.random.choice(available, p=p)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            rel_index, rel_type = self.relation(position[i], position[i + 1])

            direc[rel_index, i + 1] = 1

        return position, direc

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        xs = []
        ys = []

        for i in range(self.n_states):
            pos = i % int(self.n_states / self.n_laps)
            lap = int(i / int(self.n_states / self.n_laps))
            xs.append(pos)
            ys.append(-lap)

        if cells is not None:
            cell_prepared = np.asarray(cp.deepcopy(cells)).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


class Nieh2021(Environment):
    def __init__(self, params, length):
        self.length = length
        self.n_states = length ** 2 + 4
        super().__init__(params, length, length, self.n_states)

        # reward locations
        self.reward_pos = [self.n_states - 1, self.n_states - 4]
        # no reward locations
        self.no_reward_pos = [self.n_states - 2, self.n_states - 3]
        self.reward_pos_training = self.reward_pos + self.no_reward_pos
        self.reward_value = self.n_states / 4

        self.start_state = 0

    def world(self):
        """
        25(R) 26(NR)   27(NR) 28(R)
        16 17 18 19 20 21 22 23 24
           9  10 11 12 13 14 15
              4  5  6  7  8
                 1  2  3
                    0
        """
        length = self.length
        n_states = length ** 2 + 4
        adj = np.zeros((n_states, n_states))

        for level in range(length):
            ns = 2 * level + 1  # num states at level
            for ns_ in range(ns):
                level_add = ns_ + level ** 2
                if level == length - 1:
                    error_prob = 1 / (1 + np.exp(self.par.env.error_beta * (np.abs(ns_ - length + 1))))
                    # connect up last layer to left/right
                    if ns_ <= length - 1:
                        adj[level_add, n_states - 4] = 1  # go left
                        adj[level_add, n_states - 2] = 1 * error_prob / (1 - error_prob)  # go right = error
                    if ns_ >= length - 1:
                        adj[level_add, n_states - 3] = 1 * error_prob / (1 - error_prob)  # go left = error
                        adj[level_add, n_states - 1] = 1  # go right

                else:
                    # connect to three states above
                    adj[level_add, level_add + ns] = 1
                    adj[level_add, level_add + ns + 1] = 1
                    adj[level_add, level_add + ns + 2] = 1

        # connect left/right to start
        adj[n_states - 1, 0] = 1
        adj[n_states - 2, 0] = 1
        adj[n_states - 3, 0] = 1
        adj[n_states - 4, 0] = 1

        tran = np.zeros((n_states, n_states))
        for i in range(n_states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        adj[adj > 0] = 1  # corrects adj matrix for error_prob introduced above!

        self.adj, self.tran = adj, tran

    def relation(self, s1, s2):
        """
        26(R) 27(NR)   28(NR) 29(R)
        17 18 19 20 21 22 23 24 25
           10 11 12 13 15 15 16
              5  6  7  8  9
                 2  3  4
                    1
        """
        n_states = self.length ** 2 + 4
        diff = s2 - s1
        level = np.floor(np.sqrt(s1))
        if s2 >= n_states:
            raise ValueError('Impossible action')
        elif s2 == 0:
            rel_type = 'proceed'
        elif s2 == n_states - 1 or s2 == n_states - 2:
            rel_type = 'right'
        elif s2 == n_states - 3 or s2 == n_states - 4:
            rel_type = 'left'
        elif diff == level * 2 + 1:
            rel_type = 'pillar_left'
        elif diff == level * 2 + 3:
            rel_type = 'pillar_right'
        elif diff == level * 2 + 2:
            rel_type = 'proceed'
        else:
            raise ValueError('Impossible action')

        rel_index = self.rels.index(rel_type)
        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = np.arange(self.par.s_size)

        if self.par.use_reward:
            # choose reward sense
            reward_sense = np.random.choice(choices)
            no_reward_sense = np.random.choice(choices)
            # choices = np.delete(choices, reward_sense)
        else:
            reward_sense, no_reward_sense = 0, 0

        for i in range(self.n_states):
            # choose which sense goes where
            new_state = np.random.choice(choices)
            states_vec[i] = new_state

        # all states at same level should be the same!
        for level in range(self.width):
            for state in range(2 * level + 1):
                states_vec[level ** 2 + state] = states_vec[level ** 2]
        if self.par.use_reward:
            # make particular position special in track
            for r_p in self.reward_pos:
                states_vec[r_p] = reward_sense
            for r_p in self.no_reward_pos:
                states_vec[r_p] = no_reward_sense

        self.states_mat = states_vec.astype(int)

    def walk(self):
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        direc = np.zeros((self.n_actions, time_steps))

        position[0] = int(self.start_state)
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        direc[0, 0] = 1
        bias = []
        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)
            p = self.tran[int(position[i]), available]  # choose next position from actual allowed positions

            if position[i] == 0:
                # if at state state, choose whether more towers on left/right or evens
                # i.e. bias transitions
                bias = np.random.choice(
                    ['pillar_left', 'pillar_right'])  # , 'no_bias'])  # no_bias is not an available action
            # find relations for positions:
            rels = [self.relation(position[i], avail)[1] for avail in available]
            to_bias = bias in rels
            if to_bias:
                index = rels.index(bias)
                p[index] *= self.par.env.bias
                p = p / sum(p)

            new_poss_pos = np.random.choice(available, p=p)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            rel_index, rel_type = self.relation(position[i], position[i + 1])

            direc[rel_index, i + 1] = 1

        return position, direc

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        width = self.width
        xs = []
        ys = []

        for level in range(width):
            for j in range(level * 2 + 1):
                xs.append(j - level)
                ys.append(level)

        # left/right r/nr
        for i in range(4):
            ys.append(width + 1)
            xs.append(2 * (i - 1.5))

        if _mask is not None:
            xs = np.asarray(xs)[_mask].tolist()
            ys = np.asarray(ys)[_mask].tolist()

        cells_ = None
        if cells is not None:
            cells_ = np.asarray(cp.deepcopy(cells)).flatten()
            if _mask is not None:
                cells_ = cells_[_mask]

        if _plot_specs.nieh2021.plot_all:
            # remove final nodes
            xs = xs[:-4]
            ys = ys[:-4]
            if cells is not None:
                cells_ = cells_[:-4]

        if cells is not None:
            return xs, ys, cells_
        else:
            return xs, ys


def get_new_data_diff_envs(position, pars, envs_class):
    b_s = int(pars.batch_size)
    n_walk = position.shape[-1]  # pars.seq_len
    s_size = pars.s_size

    data = np.zeros((b_s, s_size, n_walk))
    for batch in range(b_s):
        data[batch] = sample_data(position[batch, :], envs_class[batch].states_mat, s_size)

    return data


def sample_data(position, states_mat, s_size):
    # makes one-hot encoding of sensory at each time-step
    time_steps = np.shape(position)[0]
    sense_data = np.zeros((s_size, time_steps))
    for i, pos in enumerate(position):
        ind = int(pos)
        sense_data[states_mat[ind], i] = 1
    return sense_data


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


def in_hexagon(x, y, width):
    # x, y, are centered about middle
    return np.abs(y) / np.sqrt(3) <= np.minimum(width / 4, width / 2 - np.abs(x))
