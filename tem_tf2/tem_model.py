#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import model_utils

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

eps = model_utils.eps


class TEM(tf.keras.Model):
    def __init__(self, par):
        super(TEM, self).__init__()

        self.par = par
        self.precision = tf.float32 if 'precision' not in self.par else self.par.precision
        self.mask = tf.constant(par.mask_p, dtype=self.precision, name='mask_p')
        self.mask_g = tf.constant(par.mask_g, dtype=self.precision, name='mask_g')
        self.batch_size = self.par.batch_size
        self.scalings = None  # JW: probs need to change this
        self.seq_pos = tf.zeros(self.batch_size, dtype=self.precision, name='seq_pos')
        if 'two_hot_mat' in par:
            self.two_hot_mat = tf.constant(par.two_hot_mat, dtype=self.precision, name='two_hot_mat')

        # Create trainable parameters
        glorot_uniform = tf.keras.initializers.GlorotUniform()
        trunc_norm_p2g = tf.initializers.TruncatedNormal(stddev=self.par.p2g_init)
        trunc_norm_g = tf.initializers.TruncatedNormal(stddev=self.par.g_init)

        # filtering constants
        self.gamma = [
            tf.Variable(np.log(self.par.freqs[f] / (1 - self.par.freqs[f])), dtype=self.precision, trainable=True,
                        name='gamma_' + str(f)) for f in range(self.par.n_freq)]
        # Entorhinal preference weights
        self.w_x = tf.Variable(1.0, dtype=self.precision, trainable=True, name='w_x')
        # Entorhinal preference bias
        self.b_x = tf.Variable(tf.zeros_initializer()(shape=self.par.s_size_comp, dtype=self.precision), trainable=True,
                               name='bias_x')
        # Frequency module specific scaling of sensory experience before input to hippocampus
        self.w_p = [tf.Variable(1.0, dtype=self.precision, trainable=True, name='w_p_' + str(f)) for f in
                    range(self.par.n_freq)]

        # g_prior mu
        self.g_prior_mu = tf.Variable(trunc_norm_g(shape=(1, self.par.g_size), dtype=self.precision), trainable=True,
                                      name='g_prior_mu')
        # g_prior logsig
        self.g_prior_logsig = tf.Variable(trunc_norm_g(shape=(1, self.par.g_size), dtype=self.precision),
                                          trainable=True, name='g_prior_logsig')

        self.g_init = None

        # MLP for transition weights
        self.t_vec = tf.keras.Sequential([Dense(self.par.d_mixed_size, input_shape=(self.par.n_actions,),
                                                activation=tf.tanh, kernel_initializer=glorot_uniform, name='t_vec_1',
                                                use_bias=False), Dense(self.par.g_size ** 2, use_bias=False,
                                                                       kernel_initializer=tf.zeros_initializer,
                                                                       name='t_vec_2')])

        # p2g
        if 'p' in self.par.infer_g_type:
            self.p2g_mu = [tf.keras.Sequential([Dense(2 * g_size, input_shape=(phase_size,), activation=tf.nn.elu,
                                                      name='p2g_mu_1_' + str(f), kernel_initializer=glorot_uniform),
                                                Dense(g_size, name='p2g_mu_2_' + str(f),
                                                      kernel_initializer=trunc_norm_p2g)]) for f, (g_size, phase_size)
                           in enumerate(zip(self.par.n_grids_all, self.par.n_phases_all))]

            self.p2g_logsig = [tf.keras.Sequential([Dense(2 * g_size, input_shape=(2,), activation=tf.nn.elu,
                                                          kernel_initializer=glorot_uniform,
                                                          name='p2g_logsig_1_' + str(f)),
                                                    Dense(g_size, kernel_initializer=glorot_uniform, activation=tf.tanh,
                                                          name='p2g_logsig_2_' + str(f))]) for f, g_size in
                               enumerate(self.par.n_grids_all)]

        # g2g logsigs
        self.g2g_logsig_inf = [tf.keras.Sequential([Dense(2 * g_size, input_shape=(g_size,), activation=tf.nn.elu,
                                                          kernel_initializer=glorot_uniform,
                                                          name='g2g_logsig_inf_1_' + str(f)),
                                                    Dense(g_size, activation=tf.tanh, kernel_initializer=glorot_uniform,
                                                          name='g2g_logsig_inf_2_' + str(f))]) for f, g_size in
                               enumerate(self.par.n_grids_all)]

        # MLP for compressing sensory observation
        if not self.par.two_hot:
            self.MLP_c = tf.keras.Sequential([Dense(self.par.s_size_comp_hidden, input_shape=(self.par.s_size,),
                                                    activation=tf.nn.elu, kernel_initializer=glorot_uniform,
                                                    name='MLP_c_1'),
                                              Dense(self.par.s_size_comp, kernel_initializer=glorot_uniform,
                                                    name='MLP_c_2')])

        self.MLP_c_star = tf.keras.Sequential([Dense(self.par.s_size_comp_hidden, input_shape=(self.par.s_size_comp,),
                                                     activation=tf.nn.elu, kernel_initializer=glorot_uniform,
                                                     name='MLP_c_star_1'),
                                               Dense(self.par.s_size, kernel_initializer=glorot_uniform,
                                                     name='MLP_c_star_2')])

    @model_utils.define_scope
    def call(self, inputs, training=None, mask=None):

        # inputs = model_utils.copy_tensor(inputs_)
        # Setup member variables and get dictionaries from input
        memories_dict, variable_dict = self.init_input(inputs)

        # Precompute transitions
        ta_mat = self.precomp_trans(inputs.d)

        # book-keeping
        g_t, x_t = inputs.g, inputs.x_
        for i in tf.range(self.par.seq_len, name='iteration') if self.par.tf_range else range(self.par.seq_len):
            # tf.range turns everything into tensors. Be careful with that! E.g. in mem_step use 'gen', 'inf'
            # tf.range (and tf in general) is slow with conditionals. Don't use where possible
            # using and appending to lists is slow with tf.range
            # tf.range version slower (+30%) than range version, though faster compilation (1000s to 80s for bptt=75)
            # tf.range version uses much less RAM

            # single step
            g_t, x_t, variable_dict, memories_dict = self.step(inputs, g_t, x_t, variable_dict, memories_dict, i,
                                                               ta_mat.read(i))

        # Now do full hebbian matrices update after BPTT truncation
        hebb_mat, hebb_mat_inv = self.final_hebbian(inputs.hebb_mat, inputs.hebb_mat_inv, memories_dict)

        # convert tensorarray to list
        variable_dict = self.tensorarray_2_list(variable_dict)
        variable_dict['weights']['gamma'] = self.gamma

        # Collate x_s, g for re-input to model
        re_input_dict = model_utils.DotDict({'a_rnn': hebb_mat,
                                             'a_rnn_inv': hebb_mat_inv,
                                             'x_s': tf.concat(x_t, axis=1, name='x_s_concat'),
                                             'g': g_t
                                             })

        return variable_dict, re_input_dict

    # WRAPPER FUNCTIONS

    @model_utils.define_scope
    def step(self, inputs, g_t, x_t, variable_dict, memories_dict, i, t_mat, mem_offset=0):
        # with tf.range and in graph mode, can't make the 'i' variable a global. So pass seq_pos, and i
        seq_pos = inputs.seq_i * self.par.seq_len + tf.cast(i, self.precision)

        # generative transition
        g_gen, g2g_all = self.gen_g(g_t, t_mat, seq_pos)

        # infer hippocampus (p) and entorhinal (g)
        mem_inf = self.mem_step(memories_dict, 'inf', i + mem_offset)
        g, p, x_s, p_x = self.inference(g2g_all, inputs.x[i], inputs.x_two_hot[i], x_t, mem_inf, inputs.d[i])

        # generate sensory
        mem_gen = self.mem_step(memories_dict, 'gen', i + mem_offset)
        x_all, x_logits_all, p_g = self.generation(p, g, g_gen, mem_gen)

        # Hebbian update - equivalent to the matrix updates, but implemented differently for computational ease
        memories_dict = self.hebbian(p, p_g, p_x, memories_dict, i + mem_offset)

        # Collate all variables for losses and saving representations
        var_updates = [[['p', 'p'], p],
                       [['p', 'p_g'], p_g],
                       [['p', 'p_x'], p_x],
                       [['g', 'g'], g],
                       [['g', 'g_gen'], g_gen],
                       [['x_s'], x_s],
                       [['pred', 'x_p'], x_all['x_p']],
                       [['pred', 'x_g'], x_all['x_g']],
                       [['pred', 'x_gt'], x_all['x_gt']],
                       [['logits', 'x_p'], x_logits_all['x_p']],
                       [['logits', 'x_g'], x_logits_all['x_g']],
                       [['logits', 'x_gt'], x_logits_all['x_gt']],
                       ]

        # And write all variables to tensorarrays
        variable_dict = self.update_vars(variable_dict, var_updates, i)

        return g, x_s, variable_dict, memories_dict

    @model_utils.define_scope
    def inference(self, g2g_all, x, x_two_hot, x_, memories, d):
        """
        Infer all variables
        """
        # get sensory input to hippocampus
        x2p, x_s, _, x_comp = self.x2p(x, x_, x_two_hot, d)

        # infer entorhinal
        g, p_x = self.infer_g(g2g_all, x2p, x, memories)

        # infer hippocampus
        p = self.infer_p(x2p, g)

        return g, p, x_s, p_x

    @model_utils.define_scope
    def generation(self, p, g, g_gen, memories):
        """
        Generate all variabes
        """
        x_p, x_p_logits = self.f_x(p)

        p_g = self.gen_p(g, memories)
        x_g, x_g_logits = self.f_x(p_g)

        p_gt = self.gen_p(g_gen, memories)
        x_gt, x_gt_logits = self.f_x(p_gt)

        x = model_utils.DotDict({'x_p': x_p,
                                 'x_g': x_g,
                                 'x_gt': x_gt})
        x_logits = model_utils.DotDict({'x_p': x_p_logits,
                                        'x_g': x_g_logits,
                                        'x_gt': x_gt_logits})

        return x, x_logits, p_g

    @model_utils.define_scope
    def final_hebbian(self, h_mat, h_mat_inv, memories_dict):
        """
        Wrapper for final Hebbian matrix computation
        :return:
        """
        mem_seq_len = memories_dict['gen']['a'].shape[-1]

        forget_mat = (self.scalings.forget * self.par.lambd) ** mem_seq_len
        forget_vec = (self.scalings.forget * self.par.lambd) ** tf.constant(np.arange(mem_seq_len)[::-1],
                                                                            shape=(1, 1, mem_seq_len),
                                                                            dtype=self.precision)

        mem_a = memories_dict['gen']['a'] * forget_vec
        mem_b = memories_dict['gen']['b']

        h_mat_new = h_mat * forget_mat + tf.matmul(mem_b, tf.transpose(mem_a, perm=(0, 2, 1))) * self.mask
        h_mat_new_ = tf.clip_by_value(h_mat_new, -self.par.hebb_lim, self.par.hebb_lim, name='h_mat')

        if 'p' in self.par.infer_g_type:
            mem_e = memories_dict['inf']['a'] * forget_vec
            mem_f = memories_dict['inf']['b']

            h_mat_inv_new = h_mat_inv * forget_mat + tf.matmul(mem_f, tf.transpose(mem_e, perm=(0, 2, 1)))
            h_mat_inv_new_ = tf.clip_by_value(h_mat_inv_new, -self.par.hebb_lim, self.par.hebb_lim, name='h_mat_inv')

            return h_mat_new_, h_mat_inv_new_
        else:
            return h_mat_new_, h_mat_inv

    # INFERENCE FUNCTIONS

    @model_utils.define_scope
    def infer_g(self, g2g_all, mu_x2p, x, memories):
        """
        Infer grids cells
        :param g2g_all: mean + variance from grids on previous time step
        :param mu_x2p: input to attractor from sensory data
        :param x: immediate sensory data
        :param memories: memory dict
        :return: inference grid cells
        """

        p_x = None
        mu, sigma = g2g_all

        # Inference - factorised posteriors
        if 'p' in self.par.infer_g_type:
            mu_p2g, sigma_p2g, p_x = self.p2g(mu_x2p, x, memories)
            _, mu, _, sigma = model_utils.combine2(mu, mu_p2g, sigma, sigma_p2g, self.batch_size)

        return mu, p_x

    @model_utils.define_scope
    def infer_p(self, x2p, g):
        """
        Infer place cells on basis of data as well as grid cells
        :param x2p: mean of distribution from data
        :param g: grid cell input
        :return: place cells
        """
        # grid input to hippocampus
        g2p = self.g2p(g)
        # hippocampus is conjunction between grid input and sensory input
        p = g2p * x2p

        # apply activation
        p = self.activation(p, 'p')

        return p

    @model_utils.define_scope
    def p2g(self, x2p, x, memories):
        """
        Pattern completion - can we aid our inference of where we are based on sensory data that we may have seen before
        :param x2p: input to place cells from data
        :param x: sensory input to help tell if memory retrieved well
        :param memories: memory dict
        :return: parameters of distributions, as well as terms for Hebbian update
        """

        # extract inverse memory
        p_x = self.attractor(x2p, memories)

        # sum over senses
        mu_attractor_sensum = tf.reduce_mean(
            tf.reshape(p_x, (self.batch_size, self.par.tot_phases, self.par.s_size_comp)), axis=2)

        mu_attractor_sensum_ = tf.split(mu_attractor_sensum, num_or_size_splits=self.par.n_phases_all, axis=1)

        mus = [self.p2g_mu[f](x) for f, x in enumerate(mu_attractor_sensum_)]
        mu = self.activation(tf.concat(mus, axis=1), 'g')

        # logsig based on whether memory is a good one or not - based on length of retrieved memory
        x_hat, _ = self.f_x(p_x)
        err = model_utils.squared_error(x, x_hat, keepdims=True)  # why squared error and not cross-entropy??
        err = tf.stop_gradient(err)
        logsig_input = [tf.stop_gradient(tf.concat([tf.reduce_sum(x ** 2, keepdims=True, axis=1), err], axis=1)) for x
                        in mus]

        logsigmas = [self.p2g_logsig[i](x) for i, x in enumerate(logsig_input)]
        logsigma = tf.concat(logsigmas, axis=1) * self.par.logsig_ratio + self.par.logsig_offset

        # ignore p2g at beginning when memories crap
        sigma = tf.exp(logsigma) + (1 - self.scalings.p2g_use) * self.par.p2g_sig_val

        return mu, sigma, p_x

    @model_utils.define_scope
    def x2p(self, x, x_t, x_two_hot, d):
        """
        Provides input to place cell layer from data
        :param x: immediate sensory data
        :param x_t: temporally filtered data from previous time-step
        :param x_two_hot: two-hot encoding
        :param d: current direction
        :return: input to place cell layer
        """
        if self.par.two_hot:
            # if using two hot encoding of sensory stimuli
            x_comp = x_two_hot
        else:
            # otherwise compress one-hot encoding
            x_comp = self.MLP_c(x)

        # temporally filter
        x_ = self.x2x_(x_comp, x_t, d)
        # normalise
        x_normed = self.f_n(x_)
        # tile to make same size as hippocampus
        x_2p = self.x_2p(x_normed)

        return x_2p, x_, tf.concat(x_normed, axis=1), x_comp

    @model_utils.define_scope
    def g2p(self, g):
        """
        input from grid cells to place cell layer
        :param g: grid cells
        :return: input to place cell layer
        """
        g2p_ = self.g_downsample(g)

        # repeat to get same dimension as hippocampus - same as applying W_repeat
        g2p = model_utils.tf_repeat_axis_1(g2p_, self.par.s_size_comp, self.par.p_size)

        return g2p

    @model_utils.define_scope
    def g_downsample(self, g):
        # split into frequencies
        gs = tf.split(g, num_or_size_splits=self.par.n_grids_all, axis=1)
        # down-sampling - only take a subsection of grid cells
        gs_ = [grids[:, :self.par.n_phases_all[freq]] for freq, grids in enumerate(gs)]
        g2p_ = tf.concat(gs_, axis=1)
        return g2p_

    @model_utils.define_scope
    def x2x_(self, x, x_, d):
        """
        Temporally filter data in different frequency bands
        :param x: input (compressed or otherwise
        :param x_: previous filtered values
        :param d:
        :return: new filtered values
        """

        x_s = []
        for f in range(self.par.n_freq):
            # get filtering parameter for each frequency
            # inverse sigmoid as initial parameters
            a = tf.sigmoid(self.gamma[f])

            # filter
            filtered = a * x_[f] + x * (1 - a)
            if self.par.smooth_only_on_movement:
                # only filter if actually moved
                stay_still = tf.reduce_sum(d, axis=1, keepdims=True)
                filtered = filtered * stay_still + (1 - stay_still) * x_[f]
            x_s.append(filtered)

        return x_s

    @model_utils.define_scope
    def x_2p(self, x_):
        """
        Provides input to place cell layer from filtered data
        :param x_: temporally filtered data
        :return:
        """
        # scale by w_p and tile to have appropriate place cell size (same as W_tile)
        mus = [tf.tile(tf.sigmoid(self.w_p[f]) * x_[f], (1, self.par.n_phases_all[f])) for f in
               range(self.par.n_freq)]

        mu = tf.concat(mus, 1)

        return mu

    # GENERATIVE FUNCTIONS
    @model_utils.define_scope
    def gen_p(self, g, memories):
        """
        generate place cell based on grids
        :param g: grids
        :param memories: dictionary of memory stuff
        :return:
        """

        # grid input to hippocampus
        g2p = self.g2p(g)

        # retrieve memory via attractor network
        retrieved_mem = self.attractor(g2p, memories)

        return retrieved_mem

    @model_utils.define_scope
    def gen_g(self, g, t_mat, seq_pos):
        """
        wrapper for generating grid cells from previous time step - sepatated into when for inferene and generation
        :param g:
        :param t_mat:
        :param seq_pos:
        :return:
        """

        seq_pos_ = tf.expand_dims(seq_pos, axis=1)

        # generative prior on grids if first step in environment, else transition
        mu_gen, sigma_gen = self.g2g(g, t_mat, name='gen')

        mu_prior, sigma_prior = self.g_prior()

        mu_inf_, sigma_inf_ = self.g2g(g, t_mat, name='inf')

        return tf.where(seq_pos_ > 0, mu_gen, mu_prior), [
            tf.where(seq_pos_ > 0, mu_inf_, mu_prior), tf.where(seq_pos_ > 0, sigma_inf_, sigma_prior)]

    @model_utils.define_scope
    def g2g(self, g, t_mat, name=''):
        """
        make grid to grid transisiton
        :param g: grid from previous timestep
        :param t_mat: direction of travel
        :param name: whether generative of inference
        :return:
        """

        # transition update
        update = self.get_g2g_update(g, t_mat)
        # add on update to current representation
        mu = update + g
        # apply activation
        mu = self.activation(mu, 'g')

        # get variance
        gs = tf.split(tf.stop_gradient(g), num_or_size_splits=self.par.n_grids_all, axis=1)

        if name == 'gen':
            logsig = 0.0  # [self.g2g_logsig_gen[f](x) for f, x in enumerate(gs)]
        elif name == 'inf':
            logsigs = [self.g2g_logsig_inf[f](x) for f, x in enumerate(gs)]
            logsig = tf.concat(logsigs, axis=1) * self.par.logsig_ratio + self.par.logsig_offset
        else:
            raise ValueError('Incorrect name given')

        sigma = tf.exp(logsig)

        return mu, sigma

    @model_utils.define_scope
    def g_prior(self):
        """
        Gives prior distribution for grid cells
        :return:
        """

        mu = self.g_init if self.g_init is not None else tf.tile(self.g_prior_mu, [self.batch_size, 1])
        logsig = tf.tile(self.g_prior_logsig, [self.batch_size, 1]) + self.par.logsig_offset  # JW: diff

        sigma = tf.exp(logsig)

        return mu, sigma

    @model_utils.define_scope
    def get_transition(self, d):
        # get transition matrix based on relationship / action
        t_vec = self.t_vec(d)
        # turn vector into matrix
        trans_all = tf.reshape(t_vec, [self.batch_size, self.par.g_size, self.par.g_size])
        # apply mask - i.e. if hierarchically or only transition within frequency
        return trans_all * self.mask_g

    @model_utils.define_scope
    def get_g2g_update(self, g_p, t_mat):

        # multiply current entorhinal representation by transition matrix
        update = tf.squeeze(tf.matmul(t_mat, tf.expand_dims(g_p, axis=2)))

        return update

    @model_utils.define_scope
    def f_x(self, p):
        """
        :param p: place cells
        :return: sensory predictions
        """
        ps = tf.split(value=p, num_or_size_splits=self.par.n_place_all, axis=1)

        # same as W_tile^T
        x_s = tf.reduce_sum(
            tf.reshape(ps[self.par.prediction_freq], (self.batch_size, self.par.n_phases_all[
                self.par.prediction_freq], self.par.s_size_comp)), axis=1)

        x_logits_ = self.w_x * x_s + self.b_x
        # decompress sensory
        x_logits = self.MLP_c_star(x_logits_)

        x = tf.nn.softmax(x_logits)

        return x, x_logits

    # ATTRACTOR FUNCTIONS
    @model_utils.define_scope
    def attractor(self, init, memories):
        """
        Attractor network for retrieving memories
        :param init: input to attractor
        :param memories: memory stuff
        :return: retrieved memory
        """

        # makes sure p_g, p_x has right shape for tracing - see tf.ensure_shape at end of function
        shape_p = init.shape

        p = self.activation(init, 'p')
        p_f = tf.split(p, num_or_size_splits=self.par.n_place_all, axis=1)

        for i in range(self.par.n_recurs):
            # get Hebbian update
            update = self.hebb_scal_prod(p, i, memories)

            # Do attractor step
            for f in memories['attractor_freq_iterations'][i]:
                p_f[f] = self.f_p_freq(self.par.kappa * p_f[f] + update[f], f)

            p = tf.ensure_shape(tf.concat(p_f, axis=1), shape_p)

        return p

    @model_utils.define_scope
    def hebb_scal_prod(self, p, it_num, memories):
        """
        Uses scalar products instead of explicit matrix calculations. Makes everything faster.
        Note that this 'efficient implementation' will be costly if our sequence length is greater than the hidden
        state dimensionality
        Wrapper function for actual computation of scalar products
        :param p: current state of attractor
        :param it_num: current iteration number
        :param memories: memory stuff
        :return:
        """

        p_ = tf.expand_dims(p, axis=1)
        ps = tf.split(p, num_or_size_splits=self.par.n_place_all, axis=1)

        updates_poss = self.hebb_scal_prod_helper(memories, ps, it_num)

        # when converting to graph, if using lists and conditionals -  tf.cond gets funny if list len not predefined
        updates = [tf.zeros((self.batch_size, self.par.n_place_all[freq])) for freq in range(self.par.n_freq)]

        # get Hebbian updates
        for freq in memories['attractor_freq_iterations'][it_num]:
            updates[freq] = updates_poss[freq] + \
                            tf.squeeze(tf.matmul(p_, memories['attractor_matrix'][freq])) * memories['forget_mat']

        return updates

    @model_utils.define_scope
    def hebb_scal_prod_helper(self, memories, ps, it_num):
        """
        Computations of scalar products
        :param memories: memories info
        :param ps: current state of attractor
        :param it_num: current iteration number
        :return:
        """

        scal_prods = [0.0 for _ in range(self.par.n_freq)]
        # pre-calculate scalar prods for each freq:
        for freq in range(self.par.n_freq):
            p_freq = tf.expand_dims(ps[freq], axis=2)
            scal_prods[freq] = tf.matmul(tf.transpose(memories['b_freq'][freq], (0, 2, 1)), p_freq)

        updates = [tf.zeros_like(ps[freq], dtype=self.precision) for freq in range(self.par.n_freq)]
        for freq in memories['attractor_freq_iterations'][it_num]:
            scal_prod_sum = tf.zeros_like(scal_prods[freq])

            # only use scalar prods that are needed for each freq
            for f in memories['r_f_f'][freq]:
                scal_prod_sum += scal_prods[f]

            updates[freq] = tf.squeeze(tf.matmul(memories['a_freq'][freq], scal_prod_sum * memories['forget_vec']))

        return updates

    # Memory functions
    @model_utils.define_scope
    def hebbian(self, p, p_g, p_x, mems, mem_i):
        """
        :param p: inferred place cells
        :param p_g: generated place cells
        :param p_x: retrieved memory from sensory data
        :param mems: memories dict
        :param mem_i:
        :return:

        This process is equivalent to updating Hebbian matrices, though it is more computationally efficient.
        See Ba et al 2016.
        """

        a, b = p - p_g, p + p_g
        e, f = None, None
        if self.par.hebb_type == [[2], [2]] and p_x is not None:
            # Inverse
            e, f = p - p_x, p + p_x

        # add memories to a list
        mems['gen']['a'] = self.mem_update(a, mems['gen']['a'], mem_i)
        mems['gen']['b'] = self.mem_update(b, mems['gen']['b'], mem_i)
        if e is not None and f is not None:
            mems['inf']['a'] = self.mem_update(e, mems['inf']['a'], mem_i)
            mems['inf']['b'] = self.mem_update(f, mems['inf']['b'], mem_i)

        return mems

    @model_utils.define_scope
    def mem_update(self, mem, mems, mem_num):
        """
        Update bank of memories (for scalar product computations)
        :param mem: memory to add
        :param mems: current memories
        :param mem_num:
        :return:
        """
        indices = tf.expand_dims(tf.expand_dims(mem_num, axis=0), axis=0)

        # forget all past memories (sqrt as get multiplied by another memories)
        # mems = tf.multiply(mems, tf.sqrt(self.scalings.forget * self.par.lambd))
        # add new memory - clearly shouldn't have to do two transposes
        mems = tf.transpose(mems, [2, 0, 1])
        # don't have to multiply by self.par.eta * self.scalings.h_l) here either, but I think more efficent here
        mems = tf.tensor_scatter_nd_update(mems, indices,
                                           tf.expand_dims(tf.sqrt(self.par.eta * self.scalings.h_l) * mem, axis=0))
        mems = tf.transpose(mems, [1, 2, 0])

        return mems

    # Activation functions
    @model_utils.define_scope
    def f_n(self, x):
        x_normed = []
        # apply normalisation to each frequency separately
        for f in range(self.par.n_freq):
            # subtract mean and threshold
            x_demean = tf.nn.relu(x[f] - tf.reduce_mean(x[f], axis=1, keepdims=True))
            # l2 normalise
            x_normed.append(tf.nn.l2_normalize(x_demean, axis=1))

        return x_normed

    @model_utils.define_scope
    def apply_function_freqs(self, x, act, dim):
        if isinstance(x, list):
            return [act(x[f], f) for f in range(self.par.n_freq)]
        elif isinstance(x, tf.Tensor):
            xs = tf.split(value=x, num_or_size_splits=dim, axis=1)
            # apply activation to each frequency separately
            xs = [act(xs[f], f) for f in range(self.par.n_freq)]
            return tf.concat(xs, axis=1)
        else:
            raise ValueError('in correct type given - ' + str(type(x)))

    @model_utils.define_scope
    def activation(self, x, name):
        if name == 'g':
            act = self.f_g_freq
            dim = self.par.n_grids_all
        elif name == 'p':
            act = self.f_p_freq
            dim = self.par.n_place_all
        else:
            raise ValueError('Name <' + name + '> not supported')

        return self.apply_function_freqs(x, act, dim)

    @model_utils.define_scope
    def f_g_freq(self, g, _):
        return tf.minimum(tf.maximum(g, -1), 1)

    @model_utils.define_scope
    def f_p_freq(self, p, _):
        return tf.nn.leaky_relu(tf.minimum(tf.maximum(p, -1), 1), alpha=0.01)

    @model_utils.define_scope
    def threshold(self, g):
        # make this a softer threshold - i.e. shallow gradient past threshold?
        between_thresh = tf.minimum(tf.maximum(g, self.par.thresh_min), self.par.thresh_max)
        above_thresh = tf.maximum(g, self.par.thresh_max) - self.par.thresh_max
        below_thresh = tf.minimum(g, self.par.thresh_min) - self.par.thresh_min

        return between_thresh + 0.01 * (above_thresh + below_thresh) if self.par.threshold else g

    @model_utils.define_scope
    def init_mems(self, hebb_mat, hebb_mat_inv, new_mems):
        # prespecifying everything as tf is happier when all shapes are set beforehand
        # for some reason the memories end up being immutable if DotDict. So use normal dict here
        # COULD USE TENSORARRAYS HERE TOO
        memories_dict = {'gen': {'max_attractor_its': self.par.max_attractor_its,
                                 'r_f_f': self.par.R_f_F_,
                                 'attractor_freq_iterations': self.par.attractor_freq_iterations,
                                 'attractor_matrix': hebb_mat,
                                 'a': tf.zeros((self.batch_size, self.par.p_size, new_mems)),
                                 'b': tf.zeros((self.batch_size, self.par.p_size, new_mems))
                                 },
                         'inf': {'max_attractor_its': self.par.max_attractor_its_inv,
                                 'r_f_f': self.par.R_f_F_inv_,
                                 'attractor_freq_iterations': self.par.attractor_freq_iterations_inv,
                                 'attractor_matrix': hebb_mat_inv,
                                 'a': tf.zeros((self.batch_size, self.par.p_size, new_mems)),
                                 'b': tf.zeros((self.batch_size, self.par.p_size, new_mems))
                                 }
                         }

        return memories_dict

    @model_utils.define_scope
    def mem_step(self, mems, gen_inf, itnum):

        # can streamline this. Don't need to make full dict again for example.

        mem_s = {'max_attractor_its': mems[gen_inf]['max_attractor_its'],
                 'r_f_f': self.par.R_f_F_ if gen_inf == 'gen' else self.par.R_f_F_inv_,
                 'attractor_freq_iterations': self.par.attractor_freq_iterations if gen_inf == 'gen' else
                 self.par.attractor_freq_iterations_inv,
                 'attractor_matrix': mems[gen_inf]['attractor_matrix'],
                 'a_freq': tf.split(mems[gen_inf]['a'][:, :, :itnum], num_or_size_splits=self.par.n_place_all, axis=1),
                 'b_freq': tf.split(mems[gen_inf]['b'][:, :, :itnum], num_or_size_splits=self.par.n_place_all, axis=1),
                 'forget_vec': (self.scalings.forget * self.par.lambd) ** tf.reverse(
                     tf.constant(np.arange(self.par.seq_len), dtype=self.precision, shape=(1, self.par.seq_len, 1))
                     [:, :itnum, :], axis=[1]),
                 'forget_mat': (self.scalings.forget * self.par.lambd) ** tf.cast(tf.identity(itnum), self.precision)
                 }

        return mem_s

    @model_utils.define_scope
    def init_input(self, inputs, new_mems=None):
        """
        Set model member variables from inputs and prepare memory and variable dictionaries
        """
        # Set member variables from input
        self.batch_size = inputs.x[0].shape[0]
        self.scalings = inputs.scalings
        # get hebbian matrices
        # split into frequencies for hierarchical attractor - i.e. finish attractor early for low freq memories
        hebb_mat = tf.split(inputs.hebb_mat, num_or_size_splits=self.par.n_place_all, axis=2)
        hebb_mat_inv = tf.split(inputs.hebb_mat_inv, num_or_size_splits=self.par.n_place_all, axis=2)

        # Find how many new memories will be created in this forward pass - length of input sequence by default
        new_mems = self.par.seq_len if new_mems is None else new_mems
        # Create memory and data dictionaries
        memories_dict = self.init_mems(hebb_mat, hebb_mat_inv, new_mems)
        variable_dict = self.init_vars()
        # Return dicts
        return memories_dict, variable_dict

    @model_utils.define_scope
    def init_vars(self, seq_len=None):
        """
        Collecting variables for losses, accuracies and saving. Start with all fields that can possibly be collected.
        Then when generating output in tensorarray_2_list, only stack those fields that were actually written to.
        Tensorflow annoying any wont deal with list appends with tf.range, so using TensorArray instead        
        """
        # Total number of variables collected: if not provided, default to the length of the backprop sequence
        seq_len = self.par.seq_len if seq_len is None else seq_len

        # Create dictionary with all possible data for saving
        vars_dict = model_utils.DotDict(
            {'g': {'g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_g'),
                   'g_gen': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_g_gen'),
                   },
             'p': {'p': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_p'),
                   'p_g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_p_g'),
                   'p_gt': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_p_gt'),
                   'p_x': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_p_x')
                   },
             'x_s': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_x_s'),
             'pred': {'x_p': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_x_p'),
                      'x_g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_x_g'),
                      'x_gt': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_x_gt'),
                      'd': tf.TensorArray(self.precision, size=seq_len - 1, clear_after_read=False, name='ta_d')
                      },
             'logits': {
                 'x_p': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_logit_x_p'),
                 'x_g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_logit_x_g'),
                 'x_gt': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_logit_x_gt'),
                 'd': tf.TensorArray(self.precision, size=seq_len - 1, clear_after_read=False, name='ta_logit_d')
             },
             'weights': {'gamma': 0.0}
             })

        return vars_dict

    @model_utils.define_scope
    def update_vars(self, input_dict, updates, i):
        """
        Select specific fields for writing to an output array, or by default write all used values
        """
        # Create output vars_dict, which will have the requested fields
        output_dict = {}

        # Find all keys in input dict
        all_keys = model_utils.get_all_keys(input_dict)

        # Get updated keys and values by 'transposing' updates input
        update_keys, update_vals = [list(field) for field in zip(*updates)]

        # Run through all keys. Simply copy from input dict, unless new value was specified as update
        for key in all_keys:
            # Get tensorarray from input dict
            input_val = model_utils.nested_get(input_dict, key)
            if key in update_keys:
                # If an update was provided: set update to corresponding value
                model_utils.nested_set(output_dict, key, input_val.write(i, update_vals[update_keys.index(key)]))
            else:
                # If no update was provided: simply copy the original value
                model_utils.nested_set(output_dict, key, input_val)

        # Return output dict
        return model_utils.DotDict(output_dict)

    @model_utils.define_scope
    def precomp_trans(self, dirs, seq_len=None, name=None):
        """
        Precompute transitions for provided tensor of directions
        """
        # If sequence length is not specified: use full sequence length from parameters
        seq_len = self.par.seq_len if seq_len is None else seq_len
        # alternatively could pre-compute all types of actions and then use control flow
        ta_mat = tf.TensorArray(self.precision, size=seq_len, clear_after_read=False,
                                name='t_mat' + ('' if name is None else name))
        ds = tf.unstack(dirs, axis=0)
        for j, d in enumerate(ds):
            # Get transition matrix from action/relation
            new_ta = self.get_transition(d)
            # And write transitions for this iteration to ta_mat
            ta_mat = ta_mat.write(j, new_ta)
        return ta_mat

    @model_utils.define_scope
    def tensorarray_2_list_old(self, variable_dict):
        # likely not the best way to do this...
        vars_dict = model_utils.DotDict({'g': {'g': tf.unstack(variable_dict.g.g.stack(), axis=0, name='g_unstack'),
                                               'g_gen': tf.unstack(variable_dict.g.g_gen.stack(), axis=0,
                                                                   name='g_gen_unstack'),
                                               },
                                         'p': {'p': tf.unstack(variable_dict.p.p.stack(), axis=0, name='p_unstack'),
                                               'p_g': tf.unstack(variable_dict.p.p_g.stack(), axis=0,
                                                                 name='p_g_unstack'),
                                               'p_x': tf.unstack(variable_dict.p.p_x.stack(), axis=0,
                                                                 name='p_x_unstack')
                                               },
                                         'x_s': tf.unstack(variable_dict.x_s.stack(), axis=0, name='xs_unstack'),
                                         'pred': {'x_p': tf.unstack(variable_dict.pred.x_p.stack(), axis=0,
                                                                    name='x_p_unstack'),
                                                  'x_g': tf.unstack(variable_dict.pred.x_g.stack(), axis=0,
                                                                    name='x_g_unstack'),
                                                  'x_gt': tf.unstack(variable_dict.pred.x_gt.stack(), axis=0,
                                                                     name='x_gt_unstack'),
                                                  },
                                         'logits': {
                                             'x_p': tf.unstack(variable_dict.logits.x_p.stack(), axis=0,
                                                               name='x_p_unstack'),
                                             'x_g': tf.unstack(variable_dict.logits.x_g.stack(), axis=0,
                                                               name='x_g_unstack'),
                                             'x_gt': tf.unstack(variable_dict.logits.x_gt.stack(), axis=0,
                                                                name='x_gt_unstack'),
                                         },
                                         'weights': {'gamma': self.gamma}
                                         })

        # Add action predictions, if they exist
        if 'd' in variable_dict.pred:
            # Note ['pred']['d'] instead of .pred.d: DotDict nested assignment doesn't work
            vars_dict['pred']['d'] = tf.unstack(variable_dict.pred.d.stack(), axis=0, name='d_unstack')
            vars_dict['logits']['d'] = tf.unstack(variable_dict.logits.d.stack(), axis=0, name='d_unstack')

        vars_dict.x_s = [tf.unstack(x, axis=0, name='xs_unstack_') for x in vars_dict.x_s]

        return vars_dict

    @model_utils.define_scope
    def tensorarray_2_list(self, variable_dict):
        """
        Select specific fields for writing to an output array, or by default write all used values
        """
        # If no selection of keys to write was provided: simply select all 
        keys_to_write = model_utils.get_all_keys(variable_dict)

        # Create output vars_dict, which will have the requested fields
        vars_dict = {}

        # Then set the values of vars_dict according to fields to write from the input variable dict
        for key in keys_to_write:
            # Retrieve the value of the nested key from input variable dict and stack
            value = model_utils.nested_get(variable_dict, key)
            # Convert value to list if it is a tensorarray
            if isinstance(value, tf.TensorArray):
                # Convert tensorarray to list if it was written to at least once
                value = None if value.element_shape == tf.TensorShape(None) \
                    else tf.unstack(value.stack(), axis=0, name=key[-1] + '_unstack')
            # Set the value of the nested key in the output dict
            model_utils.nested_set(vars_dict, key, value)

        # Return output dict
        return model_utils.DotDict(vars_dict)


@model_utils.define_scope
def compute_losses(model_inputs, data, trainable_variables, par):
    lx_p = 0.0
    lx_g = 0.0
    lx_gt = 0.0
    lp = 0.0
    lg = 0.0
    lp_x = 0.0
    lg_reg = 0.0
    lp_reg = 0.0

    xs = model_inputs.x
    scalings = model_inputs.scalings
    s_visited = model_inputs.s_visited
    positions = model_inputs.positions

    s_visited_ = tf.unstack(s_visited, axis=1)
    for i in range(par.seq_len):

        if par.world_type in ['loop_laps', 'splitter', 'in_out_bound', 'tank', 'splitter_grieves'] + \
                ['wood2000', 'frank2000', 'grieves2016', 'sun2020', 'nieh2021'] and par.use_reward:
            # are we at a reward state?
            # do we want to increase prediction if at no - or - rewarded state? I.e. x_mult for R and NR in splitters?
            x_mult = tf.where(
                tf.reduce_min(tf.abs(model_inputs.reward_pos - tf.expand_dims(positions[i], axis=1)), axis=1) == 0,
                model_inputs.reward_val, 1.0)

        else:
            x_mult = 1.0

        # losses for each batch
        lx_p_ = model_utils.sparse_softmax_cross_entropy_with_logits(xs[i], data.logits.x_p[i])
        lx_g_ = model_utils.sparse_softmax_cross_entropy_with_logits(xs[i], data.logits.x_g[i])
        lx_gt_ = model_utils.sparse_softmax_cross_entropy_with_logits(xs[i], data.logits.x_gt[i])

        lp_ = model_utils.squared_error(data.p.p[i], data.p.p_g[i])
        lp_x_ = model_utils.squared_error(data.p.p[i], data.p.p_x[i]) if 'lp_x' in par.which_costs else 0
        lg_ = model_utils.squared_error(data.g.g[i], data.g.g_gen[i])

        lg_reg_ = tf.reduce_sum(data.g.g[i] ** 2, axis=1)

        lp_reg_ = tf.reduce_sum(tf.abs(data.p.p[i]), axis=1)

        # don't train on any time-steps without when haven't visited that state before.
        s_vis = s_visited_[i]
        batch_vis = tf.reduce_sum(s_vis) + eps
        # normalise for bptt sequence length
        norm = 1.0 / (batch_vis * par.seq_len)

        lx_p += tf.reduce_sum(lx_p_ * s_vis * x_mult) * norm
        lx_g += tf.reduce_sum(lx_g_ * s_vis * x_mult) * norm
        lx_gt += tf.reduce_sum(lx_gt_ * s_vis * x_mult) * norm
        lp += tf.reduce_sum(lp_ * s_vis) * scalings.temp * norm
        lg += tf.reduce_sum(lg_ * s_vis) * scalings.temp * norm
        lp_x += tf.reduce_sum(lp_x_ * s_vis) * scalings.p2g_use * scalings.temp * norm

        lg_reg += tf.reduce_sum(lg_reg_ * s_vis) * par.g_reg_pen * scalings.g_cell_reg * norm
        lp_reg += tf.reduce_sum(lp_reg_ * s_vis) * par.p_reg_pen * scalings.p_cell_reg * norm

    losses = model_utils.DotDict()
    cost_all = 0
    if 'lx_gt' in par.which_costs:
        cost_all += lx_gt
        losses.lx_gt = lx_gt
    if 'lx_p' in par.which_costs:
        cost_all += lx_p
        losses.lx_p = lx_p
    if 'lx_g' in par.which_costs:
        cost_all += lx_g
        losses.lx_g = lx_g
    if 'lg' in par.which_costs:
        cost_all += lg
        losses.lg = lg
    if 'lp' in par.which_costs:
        cost_all += lp
        losses.lp = lp
    if 'lp_x' in par.which_costs:
        cost_all += lp_x
        losses.lp_x = lp_x
    if 'lg_reg' in par.which_costs:
        cost_all += lg_reg
        losses.lg_reg = lg_reg
    if 'lp_reg' in par.which_costs:
        cost_all += lp_reg
        losses.lp_reg = lp_reg
    if 'weight_reg' in par.which_costs:
        losses.weight_reg = tf.add_n(
            [tf.nn.l2_loss(v) for v in trainable_variables if 'bias' not in v.name]) * par.weight_reg_val / par.seq_len
        cost_all += losses.weight_reg

    losses.train_loss = cost_all

    return losses
