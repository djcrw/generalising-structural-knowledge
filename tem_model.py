#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import numpy as np
import tensorflow.contrib.layers as layer
from arb_functions import *

fu_co = layer.fully_connected
eps = 1e-8


class BasicInfo:
    def __init__(self, params):
        self.par = params

        self.weight_save = True

        self.mask = tf.constant(params['mask_p'], dtype=tf.float32)
        self.mask_g = tf.constant(params['mask_g'], dtype=tf.float32)

        self.x_p, self.x_g, self.x_gt = [0] * self.par['seq_len'], [0] * self.par['seq_len'], [0] * self.par[
            'seq_len']
        self.p, self.p_g = [0] * self.par['seq_len'], [0] * self.par['seq_len']
        self.g = [0] * self.par['seq_len']
        self.x_ = [0] * self.par['seq_len']
        self.accuracy_p, self.accuracy_g, self.accuracy_gt = 0, 0, 0
        self.lx_p, self.lx_g, self.lx_gt, self.lp, self.lg, self.lg_reg,  self.lp_reg, self.lp_x, self.ovc_reg = \
            0, 0, 0, 0, 0, 0, 0, 0, 0

        self.mem_list_a, self.mem_list_b, self.mem_list_e, self.mem_list_f = [], [], [], []
        self.mem_a = tf.zeros([self.par['batch_size'], self.par['p_size'], 1], dtype=tf.float32)
        self.mem_b = tf.zeros([self.par['batch_size'], self.par['p_size'], 1], dtype=tf.float32)
        self.mem_e = tf.zeros([self.par['batch_size'], self.par['p_size'], 1], dtype=tf.float32)
        self.mem_f = tf.zeros([self.par['batch_size'], self.par['p_size'], 1], dtype=tf.float32)
        self.mem_list_a_s = [[] for _ in range(self.par['n_freq'])]
        self.mem_list_b_s = [[] for _ in range(self.par['n_freq'])]
        self.mem_list_e_s = [[] for _ in range(self.par['n_freq'])]
        self.mem_list_f_s = [[] for _ in range(self.par['n_freq'])]
        self.mem_a_s = [tf.zeros([self.par['batch_size'], self.par['n_place_all'][i], 1],
                                 dtype=tf.float32) for i in range(self.par['n_freq'])]
        self.mem_b_s = [tf.zeros([self.par['batch_size'], self.par['n_place_all'][i], 1],
                                 dtype=tf.float32) for i in range(self.par['n_freq'])]
        self.mem_e_s = [tf.zeros([self.par['batch_size'], self.par['n_place_all'][i], 1],
                                 dtype=tf.float32) for i in range(self.par['n_freq'])]
        self.mem_f_s = [tf.zeros([self.par['batch_size'], self.par['n_place_all'][i], 1],
                                 dtype=tf.float32) for i in range(self.par['n_freq'])]

        return

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)"""
        # with tf.name_scope('summaries'):
        if self.weight_save:
            with tf.name_scope(name):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)

                if 'sig' in name:
                    tf.summary.scalar('max', tf.reduce_max(var))
                    tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

    def hierarchical_logsig(self, x, name, splits, sizes, trainable, concat, k=2):
        xs = x if splits == 'done' else tf.split(value=x, num_or_size_splits=splits, axis=1)
        xs = [tf.stop_gradient(x) for x in xs]

        logsigs_ = [fu_co(xs[i], k * sizes[i], activation_fn=tf.nn.elu, reuse=tf.AUTO_REUSE, scope=name + '_' + str(i),
                          weights_initializer=layer.xavier_initializer(),
                          trainable=trainable) for i in range(self.par['n_freq'])]
        logsigs = [self.par['logsig_ratio'] * fu_co(logsigs_[i], sizes[i], activation_fn=tf.nn.tanh,
                                                    reuse=tf.AUTO_REUSE, scope=name + str(i),
                                                    weights_initializer=layer.xavier_initializer(),
                                                    trainable=trainable) for i in range(self.par['n_freq'])]

        return tf.concat(logsigs, axis=1) if concat else logsigs

    def hierarchical_mu(self, x, name, splits, sizes, concat, k=2):
        # this is just for p2g
        xs = x if splits == 'done' else tf.split(value=x, num_or_size_splits=splits, axis=1)
        mus_ = [fu_co(a, k * sizes[i], activation_fn=tf.nn.elu, reuse=tf.AUTO_REUSE, scope=name + '_' + str(i),
                      weights_initializer=layer.xavier_initializer()) for i, a in enumerate(xs)]
        mus = [fu_co(a, sizes[i], activation_fn=None, reuse=tf.AUTO_REUSE, scope=name + str(i),
                     weights_initializer=tf.truncated_normal_initializer(stddev=self.par['p2g_init']))
               for i, a in enumerate(mus_)]

        return tf.concat(mus, axis=1) if concat else mus

    def get_scaling_parameters(self, index):  # these should scale with gradient updates
        temp = tf.minimum((index + 1) / self.par['temp_it'], 1)
        forget = tf.minimum((index + 1) / self.par['forget_it'], 1)
        hebb_learn = tf.minimum((index + 1) / self.par['hebb_learn_it'], 1)
        p2g_use = tf.sigmoid((index - self.par['p2g_use_it']) / self.par['p2g_scale'])  # from 0 to 1
        l_r = (self.par['learning_rate_max'] - self.par['learning_rate_min']) * (self.par['l_r_decay_rate'] ** (
                    index / self.par['l_r_decay_steps'])) + self.par['learning_rate_min']
        l_r = tf.maximum(l_r, self.par['learning_rate_min'])
        g_cell_reg = 1 - tf.minimum((index + 1) / self.par['g_reg_it'], 1)
        p_cell_reg = 1 - tf.minimum((index + 1) / self.par['p_reg_it'], 1)
        ovc_cell_reg = 1 - tf.minimum((index + 1) / self.par['ovc_reg_it'], 1)

        return temp, forget, hebb_learn, p2g_use, l_r, g_cell_reg, p_cell_reg, ovc_cell_reg

    def combine2_x2g(self, mu1, mu2, sigma1, sigma2):
        # mu2 needs to be the one coming from x

        mus1 = tf.split(mu1, axis=1, num_or_size_splits=self.par['n_grids_all'])
        mus2 = tf.split(mu2, axis=1, num_or_size_splits=self.par['n_grids_all'])
        sigmas1 = tf.split(sigma1, axis=1, num_or_size_splits=self.par['n_grids_all'])
        sigmas2 = tf.split(sigma2, axis=1, num_or_size_splits=self.par['n_grids_all'])

        mus, sigmas, logsigmas = [], [], []
        out_size = tf.shape(mu1)[1]

        for freq, (m1, m2, s1, s2, cond) in enumerate(zip(mus1, mus2, sigmas1, sigmas2, self.par['x2g_freqs'])):

            if cond:
                inv_sigma_sq1 = tf.truediv(1.0, tf.square(s1))
                inv_sigma_sq2 = tf.truediv(1.0, tf.square(s2))

                logsigma_ = -0.5 * tf.log(inv_sigma_sq1 + inv_sigma_sq2)
                sigma_ = tf.exp(logsigma_)

                mu_ = tf.square(sigma_) * (m1 * inv_sigma_sq1 + m2 * inv_sigma_sq2)
            else:
                mu_ = m1
                sigma_ = s1
                logsigma_ = tf.log(sigma_)

            mus.append(mu_)
            sigmas.append(sigma_)
            logsigmas.append(logsigma_)

        mu = tf.concat(mus, axis=1)
        sigma = tf.concat(mus, axis=1)
        logsigma = tf.concat(mus, axis=1)

        e = tf.random_normal((self.par['batch_size'], out_size), mean=0, stddev=1)

        return mu + sigma * e, mu, logsigma, sigma


class TEM(BasicInfo):
    def __init__(self, x, x_, x_perf, g, d, hmat, hmat_inv, index, seq_i, s_vis, sh, no_d, params):
        BasicInfo.__init__(self, params)

        self.shiny, self.no_direction = sh, no_d
        # get all scaling parameters - they slowly change to help network learn
        self.temp, self.forget, self.h_l, self.p2g_use, l_r, self.g_cell_reg, self.p_cell_reg,  self.ovc_cell_reg \
            = self.get_scaling_parameters(index)
        # get hebbian matrices
        self.A = hmat
        self.A_inv = hmat_inv
        # split into frequencies for hierarchical attractor - i.e. finish attractor early for low freq memories
        self.A_split = tf.split(self.A, num_or_size_splits=self.par['n_place_all'], axis=2)
        self.A_inv_split = tf.split(self.A_inv, num_or_size_splits=self.par['n_place_all'], axis=2)

        x_ = tf.split(axis=1, num_or_size_splits=self.par['n_freq'], value=x_)

        for i in range(self.par['seq_len']):
            self.seq_pos = seq_i * self.par['seq_len'] + i
            if self.par['world_type'] in ['loop_laps']:  # increase cost for reward state in Sun et al task
                self.x_mult = tf.cond(tf.equal(tf.floormod(self.seq_pos, self.par['n_states'][0]),
                                               self.par['reward_pos']), lambda: self.par['reward_value'], lambda: 1)
                self.x_mult = tf.cast(self.x_mult, dtype=tf.float32)
            else:
                self.x_mult = 1

            with tf.name_scope('Model_step_' + str(i)):
                # book-keeping
                self.weight_save = True if i == 1 else False
                self.s_vis = s_vis[i]
                g_t, x_t = (g, x_) if i == 0 else (self.g[i - 1], self.x_[i - 1])

                # generative transition
                with tf.name_scope('prev_t'):
                    g_gen, g2g_all = self.gen_g(g_t, d[i])

                # infer hippocampus (p) and entorhinal (g)
                with tf.name_scope('Inference'):
                    g, p, x_s, p_x = self.inference(g2g_all, x[i], x_perf[i], x_t)

                # generate sensory
                with tf.name_scope('Generation'):
                    x_all, x_logits_all, p_g = self.generation(p, g, g_gen)

                # Hebbian update - equivalent to the matrix updates, but implemented differently for computational ease
                with tf.name_scope('Hebbian'):
                    self.hebbian(p, p_g, p_x)

            # compute losses
            with tf.name_scope('Losses'):
                self.compute_losses(x[i], x_logits_all, g, p, g_gen, p_g, p_x, s_vis[i])

            # compute accuracies
            with tf.name_scope('Accuracies'):
                self.compute_accuracies(x[i], x_all)

            # Collate all data for saving representations
            self.p[i], self.p_g[i] = p, p_g
            self.g[i] = g
            self.x_[i] = x_s
            self.x_p[i], self.x_g[i], self.x_gt[i] = x_all

        # Now do full hebbian matrices update after BPTT truncation
        self.final_hebbian()

        with tf.name_scope('Total_Loss_and_Summaries'):
            cost_all = 0
            if 'lx_gt' in self.par['which_costs']:
                cost_all += self.lx_gt
            if 'lx_p' in self.par['which_costs']:
                cost_all += self.lx_p
            if 'lx_g' in self.par['which_costs']:
                cost_all += self.lx_g
            if 'lg' in self.par['which_costs']:
                cost_all += self.lg
            if 'lp' in self.par['which_costs']:
                cost_all += self.lp
            if 'lp_x' in self.par['which_costs']:
                cost_all += self.lp_x
                tf.summary.scalar('lp_x', self.lp_x)
            if 'lg_reg' in self.par['which_costs']:
                cost_all += self.lg_reg
                tf.summary.scalar('lg_reg', self.lg_reg)
            if 'lp_reg' in self.par['which_costs']:
                cost_all += self.lp_reg
                tf.summary.scalar('lp_reg', self.lp_reg)
            if 'ovc_reg' in self.par['which_costs']:
                cost_all += self.ovc_reg
                tf.summary.scalar('ovc_reg', self.ovc_reg)
            if 'weight_reg' in self.par['which_costs']:
                varis = tf.trainable_variables()
                self.weight_reg = tf.add_n([tf.nn.l2_loss(v) for v in varis
                                            if 'bias' not in v.name]) * self.par['weight_reg_val']
                cost_all += self.weight_reg
                tf.summary.scalar('weight_reg', self.weight_reg)

            self.weight_save = True
            self.variable_summaries(self.A, 'ARNN')
            self.variable_summaries(self.A_inv, 'ARNN_inv')

            tf.summary.scalar('l_total', cost_all)
            tf.summary.scalar('lx_p', self.lx_p)
            tf.summary.scalar('lx_g', self.lx_g)
            tf.summary.scalar('lx_gt', self.lx_gt)
            tf.summary.scalar('lp', self.lp)
            tf.summary.scalar('lg', self.lg)

            tf.summary.scalar('accuracy_p', self.accuracy_p)
            tf.summary.scalar('accuracy_g', self.accuracy_g)
            tf.summary.scalar('accuracy_gt', self.accuracy_gt)

        with tf.name_scope('params_summaries'):
            tf.summary.scalar('temp', self.temp)
            tf.summary.scalar('hebb_forget', self.forget)
            tf.summary.scalar('hebb_learn', self.h_l)
            tf.summary.scalar('p2g_use', self.p2g_use)
            tf.summary.scalar('l_r', l_r)
            tf.summary.scalar('g_cell_reg', self.g_cell_reg)
            tf.summary.scalar('p_cell_reg', self.p_cell_reg)

        self.merged = tf.summary.merge_all()

        with tf.name_scope('Train'):
            optimizer = tf.train.AdamOptimizer(l_r, beta1=0.9)

            cost_all = cost_all / self.par['seq_len']

            grads = optimizer.compute_gradients(cost_all)

            capped_grads = [(tf.clip_by_norm(grad, 2), var) if grad is not None else (grad, var) for grad, var in grads]

            self.train_op_all = optimizer.apply_gradients(capped_grads)

    # WRAPPER FUNCTIONS

    def inference(self, g2g_all, x, x_two_hot, x_):
        # infer all variables

        with tf.name_scope('infer_g'):
            # get sensory input to hippocampus
            x2p, x_s, _, x_comp = self.x2p(x, x_, x_two_hot)
            # infer entorhinal
            g, p_x = self.infer_g(g2g_all, x2p, x)

        with tf.name_scope('infer_p'):
            # infer hippocampus
            p = self.infer_p(x2p, g)

        return g, p, x_s, p_x

    def generation(self, p, g, g_gen):

        with tf.name_scope('gen_p'):
            x_p, x_p_logits = self.f_x(p)

        with tf.name_scope('gen_g'):
            p_g = self.gen_p(g)
            x_g, x_g_logits = self.f_x(p_g)

        with tf.name_scope('gen_gt'):
            p_gt = self.gen_p(g_gen)
            x_gt, x_gt_logits = self.f_x(p_gt)

        x_all = (x_p, x_g, x_gt)
        x_logits_all = (x_p_logits, x_g_logits, x_gt_logits)

        return x_all, x_logits_all, p_g

    def compute_losses(self, x, x_logits_all, g, p, g_gen, p_g, p_x, visited):
        with tf.name_scope('compute_losses'):

            x_p_logits, x_g_logits, x_gt_logits = x_logits_all

            with tf.name_scope('lx_p'):
                lx_p = sparse_softmax_cross_entropy_with_logits(x, x_p_logits) * self.x_mult

            with tf.name_scope('lx_g'):
                lx_g = sparse_softmax_cross_entropy_with_logits(x, x_g_logits) * self.x_mult

            with tf.name_scope('lx_gt'):
                lx_gt = sparse_softmax_cross_entropy_with_logits(x, x_gt_logits) * self.x_mult

            with tf.name_scope('lp'):
                lp = squared_error(p, p_g)

            with tf.name_scope('lp_x'):
                lp_x = squared_error(p, p_x) if 'lp_x' in self.par['which_costs'] else 0

            with tf.name_scope('lg'):
                lg = squared_error(g, g_gen)

            with tf.name_scope('lg_reg'):
                # only regularise grid cells not ovcs
                lg_reg = tf.add_n([tf.reduce_sum(z_g_ ** 2, 1) for i, z_g_ in
                                   enumerate(tf.split(g, axis=1, num_or_size_splits=self.par['n_grids_all'])) if
                                   not (i == self.par['ovc_module_num'] and self.par['ovc_module_use'] and
                                   'shiny' in self.par['poss_behaviours'])])

            with tf.name_scope('lp_reg'):
                lp_reg = tf.reduce_sum(tf.abs(p), 1)

            with tf.name_scope('ovc_reg'):
                ovc_reg = tf.reduce_sum(tf.square(tf.split(g, axis=1, num_or_size_splits=self.par['n_grids_all'])
                                                  [self.par['ovc_module_num']]), 1)

        with tf.name_scope('total_losses'):
            batch_vis = tf.reduce_sum(visited) + eps
            # don't train on any timesetep without when haent visited that state before.
            self.lx_p += tf.reduce_sum(lx_p * visited) / batch_vis
            self.lx_g += tf.reduce_sum(lx_g * visited) / batch_vis
            self.lx_gt += tf.reduce_sum(lx_gt * visited) / batch_vis
            self.lp += tf.reduce_sum(lp * visited) * self.temp / batch_vis
            self.lg += tf.reduce_sum(lg * visited) * self.temp / batch_vis
            self.lp_x += tf.reduce_sum(lp_x * visited) * self.p2g_use * self.temp / batch_vis

            self.lg_reg += tf.reduce_sum(lg_reg * visited) * self.par['g_reg_pen'] * self.g_cell_reg / batch_vis
            self.lp_reg += tf.reduce_sum(lp_reg * visited) * self.par['p_reg_pen'] * self.p_cell_reg / batch_vis
            self.ovc_reg += tf.reduce_sum(ovc_reg * visited) * self.par['ovc_reg_pen'] * self.ovc_cell_reg / batch_vis

        return

    def compute_accuracies(self, x, x_all):
        """
        Compute all accuracies
        :param x:
        :param x_all:
        :return:
        """

        # work out accuracy
        x_p, x_g, x_gt = x_all
        self.accuracy_p += acc_tf(x, x_p) / self.par['seq_len']  # acc of inferred
        self.accuracy_g += acc_tf(x, x_g) / self.par['seq_len']  # acc of generated
        self.accuracy_gt += acc_tf(x, x_gt) / self.par['seq_len']  # acc of generated

        return

    def final_hebbian(self):
        """
        Wrapper for final Hebbian matrix computation
        :return:
        """

        print('fin_hebb')
        for i_num, h_type in enumerate(self.par['hebb_type']):
            if i_num > 0:
                self.A_inv += tf.matmul(self.mem_f, tf.transpose(self.mem_e, [0, 2, 1]))
            else:
                self.A += tf.matmul(self.mem_b, tf.transpose(self.mem_a, [0, 2, 1]))

        self.A = tf.multiply(self.A, self.mask)
        self.A = tf.clip_by_value(self.A, -self.par['hebb_mat_max'], self.par['hebb_mat_max'])
        self.A_inv = tf.clip_by_value(self.A_inv, -self.par['hebb_mat_max'], self.par['hebb_mat_max'])
        return

    # INFERENCE FUNCTIONS

    def infer_g(self, g2g_all, mu_x2p, x):
        """
        Infer grids cells
        :param g2g_all: mean + variance from grids on previous time step
        :param mu_x2p: input to attractor from sensory data
        :param x: immediate sensory data
        :return: inference grid cells
        """

        p_x = None
        mu, sigma = g2g_all

        # Inference - factorised posteriors
        if 'p' in self.par['infer_g_type']:
            mu_p2g, sigma_p2g, p_x = self.p2g(mu_x2p, x)
            _, mu, _, sigma = combine2(mu, mu_p2g, sigma, sigma_p2g, self.par['batch_size'])

        if 'x' in self.par['infer_g_type']:
            mu_x2g, sigma_x2g = self.x2g(x)
            _, mu, _, _ = self.combine2_x2g(mu, mu_x2g, sigma, sigma_x2g)

        self.variable_summaries(mu, 'mu_g')

        return mu, p_x

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
        p = self.par['p_activation'](p)

        self.variable_summaries(p, 'p')

        return p

    def p2g(self, x2p, x):
        """
        Pattern completion - can we aid our inference of where we are based on sensory data that we may have seen before
        :param x2p: input to place cells from data
        :param x: sensory input to help tell if memory retrieved well
        :return: parameters of distributions, as well as terms for Hebbian update
        """

        with tf.name_scope('p2g'):

            # extract inverse memory
            p_x = self.attractor(x2p, self.par['which_way'][1])

            # check if memory good - does it predict data?
            x_hat, x_hat_logits = self.f_x(p_x)
            err = squared_error(x, x_hat, keep_dims=True)
            err = tf.stop_gradient(err)

            # make grid prediction. sum over senses first
            mu_attractor_sensum = tf.reduce_mean(tf.reshape(p_x, (self.par['batch_size'], self.par['tot_phases'],
                                                                  self.par['s_size_comp'])), 2)

            mus = self.hierarchical_mu(mu_attractor_sensum, 'mu_p2g', self.par['n_phases_all'], self.par['n_grids_all'],
                                       concat=False)
            mu = tf.concat(mus, axis=1)
            mu = self.f_g(mu)

            # logsig based on whether memory is a good one or not - based on length of retrieved memory
            logsig_input = [tf.concat([tf.reduce_sum(x ** 2, keepdims=True, axis=1), err], axis=1) for x in mus]

            logsigma = self.hierarchical_logsig(logsig_input, 'sig_p2g', 'done', self.par['n_grids_all'],
                                                self.par['train_sig_p2g'], concat=True, k=2)

            logsigma += self.par['logsig_offset']
            sigma = tf.exp(logsigma)

            # ignore p2g at beginning when memories crap
            sigma += (1 - self.p2g_use) * self.par['p2g_sig_val']

            self.variable_summaries(mu, 'mu_p2g')
            if self.par['train_sig_p2g']:
                self.variable_summaries(sigma, 'sig_p2g')

        return mu, sigma, p_x

    def x2g(self, x):
        """
        Data straight to grids - for used when shiny objects present
        :param x: immediate sensory data
        :return: input ot grids
        """

        with tf.name_scope('x2g'):

            x = tf.expand_dims(tf.reduce_sum(x * self.shiny, 1), axis=1)
            mus, logsigs = [0] * self.par['n_freq'], [0] * self.par['n_freq']

            for freq, (g_size, cond) in enumerate(zip(self.par['n_grids_all'], self.par['x2g_freqs'])):

                if cond:
                    intermediate = fu_co(x, 3 * g_size, activation_fn=tf.nn.elu, reuse=tf.AUTO_REUSE,
                                         scope='mu_x2g' + str(freq))
                    mus[freq] = fu_co(intermediate, g_size, activation_fn=None, reuse=tf.AUTO_REUSE,
                                      scope='mu_x2g_2' + str(freq),
                                      weights_initializer=tf.truncated_normal_initializer(stddev=self.par['x2g_init']))

                    # we want mu to be positive
                    mus[freq] = tf.abs(mus[freq])

                else:
                    mus[freq] = tf.zeros((self.par['batch_size'], g_size))

                logsigs[freq] = self.par['logsig_ratio'] * tf.layers.dense(x, g_size, activation=tf.nn.tanh,
                                                                           reuse=tf.AUTO_REUSE, use_bias=False,
                                                                           name='sig_x2g' + str(freq),
                                                                           kernel_initializer=tf.zeros_initializer,
                                                                           trainable=self.par['train_sig_x2g']) \
                    if cond else tf.zeros((self.par['batch_size'], g_size))

            mu = tf.concat(mus, axis=1)
            logsig = tf.concat(logsigs, axis=1)

            mu = self.f_g(mu)

            logsig += self.par['logsig_offset']
            sigma = tf.exp(logsig)

            self.variable_summaries(mu, 'mu_x2g')
            if self.par['train_sig_x2g']:
                self.variable_summaries(sigma, 'sig_x2g')

        return mu, sigma

    def x2p(self, x, x_t, x_two_hot):
        """
        Provides input to place cell layer from data
        :param x: immediate sensory data
        :param x_t: temporally filtered data from previous time-step
        :param x_two_hot: two-hot encoding
        :return: input to place cell layer
        """
        with tf.name_scope('x2p'):
            if self.par['two_hot']:
                # if using two hot encoding of sensory stimuli
                x_comp = x_two_hot
            else:
                # otherwise compress one-hot encoding
                x_comp = self.f_compress(x)

            # temporally filter
            x_ = self.x2x_(x_comp, x_t)
            # normalise
            x_normed = self.f_n(x_)
            # tile to make same size as hippocampus
            x_2p = self.x_2p(x_normed)

        return x_2p, x_, tf.concat(x_normed, axis=1), x_comp

    def g2p(self, g):
        """
        input from grid cells to place cell layer
        :param g: gric cells
        :return: input to place cell layer
        """
        with tf.name_scope('g2p'):
            # split into frequencies
            gs = tf.split(value=g, num_or_size_splits=self.par['n_grids_all'], axis=1)
            # down-sampling - only take a subsection of grid cells
            gs_ = [tf.slice(ting, [0, 0], [self.par['batch_size'], self.par['n_phases_all'][freq]]) for freq, ting
                   in enumerate(gs)]
            g_ = tf.concat(gs_, axis=1)
            # repeat to get same dimension as hippocampus - same as applying W_repeat
            g2p = tf_repeat_axis_1(g_, self.par['s_size_comp'], self.par['p_size'])

        return g2p

    def f_compress(self, x):
        """
        Compresses data
        :param x: input to be compressed
        :return:  compressed input representation
        """

        with tf.name_scope("f_compress"):
            x_hidden = fu_co(x, self.par['s_size_comp_hidden'], activation_fn=tf.nn.elu, reuse=tf.AUTO_REUSE,
                             scope='f_compress_1')
            x_compressed = fu_co(x_hidden, self.par['s_size_comp'], activation_fn=None, reuse=tf.AUTO_REUSE,
                                 scope='f_compress_2')

        return x_compressed

    def x2x_(self, x, x_):
        """
        Temporally filter data in different frequency bands
        :param x: input (compressed or otherwise
        :param x_: previous filtered values
        :return: new filtered values
        """

        with tf.name_scope('x2x_'):
            x_s = [0] * self.par['n_freq']
            for i in range(self.par['n_freq']):
                # get filtering parameter for each frequency
                with tf.variable_scope("x2x_" + str(i), reuse=tf.AUTO_REUSE):
                    gamma = tf.get_variable("w_smooth_freq", [1], initializer=tf.constant_initializer(
                        np.log(self.par['freqs'][i] / (1 - self.par['freqs'][i]))),
                                            trainable=True)
                    # inverse sigmoid as initial parameters
                a = tf.sigmoid(gamma)

                if self.weight_save:
                    tf.summary.scalar('smoothing', tf.reduce_mean(a))
                # filter
                x_s[i] = a * x_[i] + x * (1 - a)

        return x_s

    def x_2p(self, x_):
        """
        Provides input to place cell layer from filtered data
        :param x_: temporally filtered data
        :return:
        """

        with tf.name_scope('x_2p'):
            mus = [0] * self.par['n_freq']

            for i in range(self.par['n_freq']):

                with tf.variable_scope("x_2p" + str(i), reuse=tf.AUTO_REUSE):
                    w_p = tf.get_variable("w_p", [1], initializer=tf.constant_initializer(1.0))
                w_p = tf.sigmoid(w_p)

                # tile to have appropriate size - same as W_tile
                mus[i] = tf.tile(w_p * x_[i], (1, self.par['n_phases_all'][i]))

            mu = tf.concat(mus, 1)

            self.variable_summaries(mu, 'mu_x_2p')

        return mu

    # GENERATIVE FUNCTIONS

    def gen_p(self, g):
        """
        generate place cell based on grids
        :param g: grids
        :return:
        """

        with tf.name_scope('gen_p'):
            # grid input to hippocampus
            g2p = self.g2p(g)
            # retrieve memory via attractor network
            retrieved_mem = self.attractor(g2p, self.par['which_way'][0])

        return retrieved_mem

    def gen_g(self, g, d):
        """
        wrapper for generating grid cells from previous time step - sepatated into when for inferene and generation
        :param g:
        :param d:
        :return:
        """

        # generative prior on grids if first step in environment, else transition
        mu, sigma = tf.cond(self.seq_pos > 0, true_fn=lambda: self.g2g(g, d, self.par['no_direc_gen'], name='gen'),
                            false_fn=lambda: self.g_prior())

        # the same but for used for inference network
        mu_inf, sigma_inf = tf.cond(self.seq_pos > 0, true_fn=lambda: self.g2g(g, d, False, name='inf'),
                                    false_fn=lambda: self.g_prior())

        return mu, (mu_inf, sigma_inf)

    def g2g(self, g, d, no_direc=False, name=''):
        """
        make grid to grid transisiton
        :param g: grid from previous timestep
        :param d: direction of travel
        :param name: whether generative of inference
        :param no_direc: special parameter for no direction information when using shiny objects
        :return:
        """

        with tf.name_scope('g2g'):
            # transition update
            update = self.get_g2g_update(g, d, no_direc, name='')
            # add on update to current representation
            mu = update + g
            # apply activation
            mu = self.f_g(mu)
            # get variance
            logsig = self.hierarchical_logsig(g, 'sig_g2g' + name, self.par['n_grids_all'], self.par['n_grids_all'],
                                              self.par['train_sig_g2g'], concat=True)
            logsig += self.par['logsig_offset']

            sigma = tf.exp(logsig)

            self.variable_summaries(mu, 'mus_g2g' + name)
            if self.par['train_sig_g2g']:
                self.variable_summaries(sigma, 'sigs_g2g' + name)

        return mu, sigma

    def g_prior(self, name=''):
        """
        Gives prior distribution for grid cells
        :return:
        """
        with tf.name_scope('g_prior'):
            with tf.variable_scope("g_prior", reuse=tf.AUTO_REUSE):
                mu = tf.tile(tf.get_variable("mu_g_prior" + name, [1, self.par['g_size']],
                                             initializer=tf.truncated_normal_initializer(stddev=self.par['g_init'])),
                             [self.par['batch_size'], 1])
                logsig = tf.tile(tf.get_variable("logsig_g_prior" + name, [1, self.par['g_size']],
                                                 initializer=tf.truncated_normal_initializer(stddev=self.par['g_init'])
                                                 ), [self.par['batch_size'], 1])

            sigma = tf.exp(logsig)

        return mu, sigma

    def get_transition(self, d, name=''):
        # get transition matrix based on relationship / action
        d_mixed = fu_co(d, (self.par['d_mixed_size']), activation_fn=tf.tanh, reuse=tf.AUTO_REUSE,
                        scope='d_mixed_g2g' + name) if self.par['d_mixed'] else d

        t_vec = tf.layers.dense(d_mixed, self.par['g_size'] ** 2, activation=None, reuse=tf.AUTO_REUSE,
                                name='mu_g2g' + name, kernel_initializer=tf.zeros_initializer, use_bias=False)
        # turn vector into matrix
        trans_all = tf.reshape(t_vec, [self.par['batch_size'], self.par['g_size'], self.par['g_size']])
        # apply mask - i.e. if hierarchically or only transition within frequency
        trans_all = tf.multiply(trans_all, self.mask_g)

        return trans_all

    def get_g2g_update(self, g_p, d, no_direc, name=''):
        # get transition matrix
        t_mat = self.get_transition(d, name)
        # multiply current entorhinal representation by transition matrix
        update = tf.reshape(tf.matmul(t_mat, tf.reshape(g_p, [self.par['batch_size'], self.par['g_size'], 1])),
                            [self.par['batch_size'], self.par['g_size']])

        if no_direc:
            # directionless transition weights - used in OVC environments
            with tf.variable_scope("g2g_directionless_weights" + name, reuse=tf.AUTO_REUSE):
                t_mat_2 = tf.get_variable("g2g" + name, [self.par['g_size'], self.par['g_size']])
                t_mat_2 = tf.multiply(t_mat_2, self.mask_g)

            update = tf.where(self.no_direction > 0.5, x=tf.matmul(g_p, t_mat_2), y=update)

        return update

    def f_x(self, p):
        """
        :param p: place cells
        :return: sensory predictions
        """
        with tf.name_scope('f_x_'):

            ps = tf.split(value=p, num_or_size_splits=self.par['n_place_all'], axis=1)

            # same as W_tile^T
            x_s = tf.reduce_sum(tf.reshape(ps[self.par['prediction_freq']], (self.par['batch_size'],
                                self.par['n_phases_all'][self.par['prediction_freq']],  self.par['s_size_comp'])), 1)

            with tf.variable_scope("f_x", reuse=tf.AUTO_REUSE):
                w_x = tf.get_variable("w_x", [1], initializer=tf.constant_initializer(1.0))
                b_x = tf.get_variable("bias", [self.par['s_size_comp']], initializer=tf.constant_initializer(0.0))

            x_logits = w_x * x_s + b_x
            # decompress sensory
            x_logits = self.f_decompress(x_logits)

            x = tf.nn.softmax(x_logits)

        return x, x_logits

    def f_decompress(self, x_compressed):
        """
        Decompress x
        :param x_compressed:
        :return: decompressed version of x
        """

        with tf.name_scope("f_decompress"):
            x_hidden = fu_co(x_compressed, self.par['s_size_comp_hidden'], activation_fn=tf.nn.elu, reuse=tf.AUTO_REUSE,
                             scope='f_decompress_1')
            x = fu_co(x_hidden, self.par['s_size'], activation_fn=None, reuse=tf.AUTO_REUSE, scope='f_decompress_2')

        return x

    # ATTRACTOR FUNCTIONS

    def attractor(self, init, which_way):
        """
        Attractor network for retrieving memories
        :param init: input to attractor
        :param which_way: whether attractor being in inference (normal) or generation (inverse)
        :return: retrieved memory
        """

        if which_way == 'normal':
            hebb_diff_freq_its_max = self.par['Hebb_diff_freq_its_max']
        else:
            hebb_diff_freq_its_max = self.par['Hebb_inv_diff_freq_its_max']

        with tf.name_scope('attractor'):
            ps = [0] * (self.par['n_recurs'] + 1)
            ps[0] = self.par['p_activation'](init)

            for i in range(self.par['n_recurs']):
                # get Hebbian update
                update = self.hebb_scal_prod(ps[i], i, which_way, hebb_diff_freq_its_max)

                ps_f = tf.split(value=ps[i], num_or_size_splits=self.par['n_place_all'], axis=1)
                for f in range(self.par['n_freq']):
                    if i < hebb_diff_freq_its_max[f]:
                        ps_f[f] = self.par['p_activation'](self.par['prev_p_decay'] * ps_f[f] + update[f])
                ps[i + 1] = tf.concat(ps_f, axis=1)

            p = ps[-1]

        return p

    def hebb_scal_prod(self, p, it_num, which_way, hebb_diff_freq_its_max):
        """
        Uses scalar products instead of explicit matrix calculations. Makes everything faster.
        Note that this 'efficient implementation' will be costly if our sequence length is greater than the hidden
        state dimensionality
        Wrapper function for actual computation of scalar products
        :param p: current state of attractor
        :param it_num: current iteration number
        :param which_way: attractor being used for inference or generation
        :param hebb_diff_freq_its_max: maximum iteration of that frequency in attractor
        :return:
        """

        if which_way == 'normal':
            h_split = self.A_split
            a, b = self.mem_a_s, self.mem_b_s
            r_f_f = self.par['R_f_F']
        else:
            h_split = self.A_inv_split
            a, b = self.mem_e_s, self.mem_f_s
            r_f_f = self.par['R_f_F_inv']

        with tf.name_scope('hebb_scal_prod'):

            p_ = tf.transpose(tf.expand_dims(p, axis=2), [0, 2, 1])
            ps = tf.split(value=p, num_or_size_splits=self.par['n_place_all'], axis=1)

            updates = [0] * self.par['n_freq']
            updates_poss = self.hebb_scal_prod_helper(a, b, ps, r_f_f)

            for freq in range(self.par['n_freq']):
                # different num of iterations per freq i.e. more for higher freqs
                if it_num < hebb_diff_freq_its_max[freq] and np.sum(r_f_f, 0)[freq] > 0:
                    hebb_add = tf.squeeze(tf.matmul(p_, h_split[freq]))
                    updates[freq] = updates_poss[freq] + hebb_add

        return updates

    def hebb_scal_prod_helper(self, a, b, ps, r_f_f):
        """
        Computations of scalar products
        :param a: memories
        :param b: memories
        :param ps: current state of attractor
        :param r_f_f: matrix that says whether frequency f influences frequency F in the attractor network
        :return:
        """

        with tf.name_scope('hebb_scal_prod_helper'):

            updates = [0] * self.par['n_freq']
            scal_prods = []
            # pre-calculate scalar prods for each freq:
            for freq in range(self.par['n_freq']):
                p_freq = tf.expand_dims(ps[freq], 2)
                scal_prods.append(tf.matmul(tf.transpose(b[freq], [0, 2, 1]), p_freq))

            for freq in range(self.par['n_freq']):  # go downwards - more computationally efficient

                scal_prod = []
                for f in range(self.par['n_freq']):  # which freqs influence which other freqs

                    if r_f_f[f][freq] > 0:

                        scal_prod.append(scal_prods[f])

                scal_prod_sum = tf.add_n(scal_prod)
                updates[freq] = tf.squeeze(tf.matmul(a[freq], scal_prod_sum))

        return updates

    # Memory functions

    def hebbian(self, p, p_g, p_x):
        """
        :param p: inferred place cells
        :param p_g: generated place cells
        :param p_x: retrieved memory from sensory data
        :return:

        This process is equivalent to updating Hebbian matrices, though it is more computationally efficient.
        See Ba et al 2016.
        """

        with tf.name_scope('hebbian'):

            a, b = p - p_g, p + p_g
            e, f = None, None
            if self.par['hebb_type'] == [[2], [2]]:
                # Inverse
                e, f = p - p_x, p + p_x

            # add memories to a list
            self.mem_a, self.mem_a_s, self.mem_list_a, self.mem_list_a_s = self.mem_update(a, self.mem_list_a,
                                                                                           self.mem_list_a_s)
            self.mem_b, self.mem_b_s, self.mem_list_b, self.mem_list_b_s = self.mem_update(b, self.mem_list_b,
                                                                                           self.mem_list_b_s)
            if e is not None and f is not None:
                self.mem_e, self.mem_e_s, self.mem_list_e, self.mem_list_e_s = self.mem_update(e, self.mem_list_e,
                                                                                               self.mem_list_e_s)
                self.mem_f, self.mem_f_s, self.mem_list_f, self.mem_list_f_s = self.mem_update(f, self.mem_list_f,
                                                                                               self.mem_list_f_s)
            # 'forget' the Hebbian matrices
            self.A = self.A * self.forget * self.par['lambd']
            self.A_split = tf.split(self.A, num_or_size_splits=self.par['n_place_all'], axis=2)

            self.A_inv = self.A_inv * self.forget * self.par['lambd']
            self.A_inv_split = tf.split(self.A_inv, num_or_size_splits=self.par['n_place_all'], axis=2)

        return

    def mem_update(self, mem, mem_list, mem_list_s):
        """
        Update bank of memories (for scalar product computations)
        :param mem: memory to add
        :param mem_list: current memory list
        :param mem_list_s: current memory list split into freqs
        :return:
        """

        with tf.name_scope('mem_update'):

            # sqrt as they are multiplied by themselves
            mem_list.append(tf.multiply(tf.sqrt(self.par['eta'] * self.h_l), mem))
            for i, el in enumerate(mem_list):
                if i < len(mem_list) - 1:
                    mem_list[i] = el * tf.sqrt(self.forget * self.par['lambd'])
            mems = tf.stack(mem_list, axis=2)

            # doing it for hierarchy
            mems_s = []
            mem_s = tf.split(value=mem, num_or_size_splits=self.par['n_place_all'], axis=1)
            for i in range(self.par['n_freq']):
                mem_list_s[i].append(tf.multiply(tf.sqrt(self.par['eta'] * self.h_l), mem_s[i]))
                for j, el in enumerate(mem_list_s[i]):
                    if j < len(mem_list_s[i]) - 1:
                        mem_list_s[i][j] = el * tf.sqrt(self.forget * self.par['lambd'])
                mems_s.append(tf.stack(mem_list_s[i], axis=2))

        return mems, mems_s, mem_list, mem_list_s

    # Activation functions

    def f_n(self, x):
        with tf.name_scope('f_n'):
            x_normed = [0] * self.par['n_freq']
            for i in range(self.par['n_freq']):
                # apply normalisation to each frequency separately
                with tf.variable_scope("f_n" + str(i), reuse=tf.AUTO_REUSE):
                    x_normed[i] = x[i]
                    # subtract mean and threshold
                    x_normed[i] = tf.maximum(x_normed[i] - tf.reduce_mean(x_normed[i], axis=1, keepdims=True), 0)
                    # l2 normalise
                    x_normed[i] = tf.nn.l2_normalize(x_normed[i], axis=1)

        return x_normed

    def f_g(self, g):
        with tf.name_scope('f_g'):
            gs = tf.split(value=g, num_or_size_splits=self.par['n_grids_all'], axis=1)
            for i in range(self.par['n_freq']):
                # apply activation to each frequency separately
                gs[i] = self.f_g_freq(gs[i], i)

            g = tf.concat(gs, axis=1)

        return g

    def f_g_freq(self, g, freq):

        if self.par['ovc_module_use'] and 'shiny' in self.par['poss_behaviours'] and freq == self.par['ovc_module_num']:
            g = self.par['ovc_activation'](g)
        else:
            g = self.par['g2g_activation'](g)

        return g
