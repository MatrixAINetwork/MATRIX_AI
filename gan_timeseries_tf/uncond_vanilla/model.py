"""
Copyright 2018 The Matrix Authors
This file is part of the Matrix library.

The Matrix library is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Matrix library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the Matrix library. If not, see <http://www.gnu.org/licenses/>.
@author: Steve Deng
"""

import tensorflow as tf
import os
import math

from lib.ops import *

def optimizer(loss, var_list, learning_rate=0.001, beta1=0.9):
    step = tf.Variable(0, trainable=False)
    with tf.variable_scope('optimizer'):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)\
            .minimize(loss=loss, var_list=var_list, global_step=step)
        return opt

class GAN(object):
    def __init__(self,
                 z_dim, img_h, img_w, c_dim,
                 g_learning_rate, d_learning_rate,
                 g_beta1, d_beta1,
                 g_hidden_units, d_hidden_units):

        # initialize batch normalization
        self.z_dim = z_dim
        self.img_h = img_h
        self.img_w = img_w
        self.c_dim = c_dim
        self.g_learning_rate = g_learning_rate
        self.d_learning_rate = d_learning_rate
        self.g_beta1 = g_beta1
        self.d_beta1 = d_beta1
        self.g_hidden_units = g_hidden_units
        self.d_hidden_units = d_hidden_units

        # set placeholder

        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='noise')
        self.x = tf.placeholder(tf.float32, shape=[None, self.img_h, self.img_w, self.c_dim], name='real_data')

        self.G = self.generator(self.z, reuse=False)
        self.D_real, self.D_real_logits = self.discriminator(self.x, reuse=False)
        self.D_fake, self.D_fake_logits = self.discriminator(self.G, reuse=True)

        # calculate loss
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.D_real_logits, labels=tf.ones_like(self.D_real)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake)))

        # get trainable variables
        var_list = tf.trainable_variables()
        self.d_vars = [v for v in var_list if v.name.startswith('D/')]
        self.g_vars = [v for v in var_list if v.name.startswith('G/')]

        # set optimizer
        self.d_opt = optimizer(self.d_loss, self.d_vars, self.d_learning_rate, self.d_beta1)
        self.g_opt = optimizer(self.g_loss, self.g_vars, self.g_learning_rate, self.g_beta1)

        # other tensors
        self.saver = tf.train.Saver()
        self.sampler = self.sampler(self.z)
        self.d_features = self.features_discriminator(self.x)

    def discriminator(self, x, reuse=False, training=True, with_features=False):
        with tf.variable_scope('D', reuse=reuse):
            h0 = tf.layers.dense(x, self.d_hidden_units, tf.nn.relu, name='d0')
            logits = tf.layers.dense(h0, 1, name='d1')
            prob = tf.nn.sigmoid(logits)
            if with_features:
                return prob, logits, h0
            else:
                return prob, logits


    def generator(self, z, reuse=False, training=True):
        with tf.variable_scope("G", reuse=reuse) as scope:
            h0 = tf.layers.dense(z, self.g_hidden_units, tf.nn.relu, name='g0')
            h1 = tf.layers.dense(h0, self.img_h, name='g1')
            h1 = tf.reshape(h1, [-1, h1.get_shape()[1].value, 1, 1])
            return h1

    def features_discriminator(self, x):
        _, _, features = self.discriminator(x, reuse=True, training=False,
                                            with_features=True)
        return features

    def sampler(self, z):
        return self.generator(z, reuse=True, training=False)

    def save(self, sess, dir_checkpoint, step, model_name):
        dir_save = os.path.join(dir_checkpoint, model_name)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        prefix = os.path.join(dir_save, "model.ckpt")
        self.saver.save(sess, prefix, global_step=step)

    def load(self, sess, dir_checkpoint, model_name):
        import re
        print("[*]  Reading checkpoints ...")
        dir_load = os.path.join(dir_checkpoint, model_name)
        ckpt = tf.train.get_checkpoint_state(dir_load)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(dir_load, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
