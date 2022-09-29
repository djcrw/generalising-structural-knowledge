#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import tensorflow as tf
eps = 1e-8


def combine2(mu1, mu2, sigma1, sigma2, batch_size):
    out_size = tf.shape(mu1)[1]
    inv_sigma_sq1 = tf.truediv(1.0, tf.square(sigma1))
    inv_sigma_sq2 = tf.truediv(1.0, tf.square(sigma2))

    logsigma = -0.5 * tf.log(inv_sigma_sq1 + inv_sigma_sq2)
    sigma = tf.exp(logsigma)

    mu = tf.square(sigma) * (mu1 * inv_sigma_sq1 + mu2 * inv_sigma_sq2)
    e = tf.random_normal((batch_size, out_size), mean=0, stddev=1)
    return mu + sigma * e, mu, logsigma, sigma


def squared_error(t, o, keep_dims=False):
    return 0.5 * tf.reduce_sum(tf.square(t - o), 1, keepdims=keep_dims)


def sparse_softmax_cross_entropy_with_logits(labels, logits):
    labels = tf.argmax(labels, 1)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)


def acc_tf(real, pred):
    correct_prediction = tf.equal(tf.argmax(real, 1), tf.argmax(pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return tf.cast(accuracy * 100, tf.int32)


def bias_variable(shape, name, reuse, train=True, offset=0.0):
    with tf.variable_scope(name, reuse=reuse):
        return tf.get_variable("b", shape, initializer=tf.constant_initializer(offset), trainable=train)


def tf_repeat_axis_1(tensor, repeat, dim1):
    dim0 = tf.shape(tensor)[0]
    return tf.reshape(tf.tile(tf.reshape(tensor, (-1, 1)), (1, repeat)), (dim0, dim1))
