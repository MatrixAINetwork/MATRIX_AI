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

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class GAN(object):
    def __init__(self,
                 z_dim, img_h, img_w, c_dim,
                 g_learning_rate, d_learning_rate,
                 g_beta1, d_beta1,
                 gf_dim=64, df_dim=64):

        # initialize batch normalization
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.z_dim = z_dim
        self.img_h = img_h
        self.img_w = img_w
        self.c_dim = c_dim
        self.g_learning_rate = g_learning_rate
        self.d_learning_rate = d_learning_rate
        self.g_beta1 = g_beta1
        self.d_beta1 = d_beta1
        self.gf_dim = gf_dim
        self.df_dim = df_dim

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

    def discriminator(self, x, reuse=False, training=True, with_hiddens=False):
        with tf.variable_scope('D', reuse=reuse):
            h0 = lrelu(conv2d(x, self.df_dim, name='d_h0'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'), training))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'),training))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv'),training))
            flat = flatten(h3)
            logits = linear(flat, 1, 'd_h4_lin')
            prob = sigmoid(logits)

            if with_hiddens:
                hiddens = [h0, h1, h2, h3]
                return prob, logits, hiddens
            else:
                return prob, logits


    def generator(self, z, reuse=False, training=True):
        with tf.variable_scope("G", reuse=reuse) as scope:
            s_h, s_w = self.img_h, self.img_w
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w,2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            batch_size = tf.shape(z)[0]

            z_ = linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin')
            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim*8])
            h0 = relu(self.g_bn0(h0, training))

            h1 = deconv2d(h0,[batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = relu(self.g_bn1(h1, training))

            h2 = deconv2d(h1, [batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
            h2 = relu(self.g_bn2(h2, training))

            h3 = deconv2d(h2, [batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
            h3 = relu(self.g_bn3(h3, training))

            h4 = deconv2d(h3, [batch_size, s_h, s_w, self.c_dim], name='g_h4')
            #out = tf.nn.sigmoid(h4)

            return h4

    def features_discriminator(self, x):
        _, _, hiddens = self.discriminator(x, reuse=True, training=False,
                                            with_hiddens=True)
        h0, h1, h2, h3 = hiddens
        dim_0 = h0.get_shape()[1].value
        dim_1 = h1.get_shape()[1].value
        dim_2 = h2.get_shape()[1].value
        dim_3 = h3.get_shape()[1].value
        # f0 = flatten(tf.nn.max_pool(h0, [1, min(32, dim), 1, 1], [1, 1, 1,1],'VALID'))
        # f1 = flatten(tf.nn.max_pool(h1, [1, min(16, dim), 1, 1], [1, 1, 1,1],'VALID'))
        # f2 = flatten(tf.nn.max_pool(h2, [1, min(16, dim), 1, 1], [1, 1, 1,1],'VALID'))
        # f3 = flatten(tf.nn.max_pool(h3, [1, min(16, dim), 1, 1], [1, 1, 1,1], 'VALID'))
        # features = tf.concat([f0, f1, f2, f3], axis=1)

        # f1 = flatten(tf.nn.max_pool(h1, [1, min(32, dim_1), 1, 1], [1, 1, 1, 1], 'VALID'))
        # f2 = flatten(tf.nn.max_pool(h2, [1, min(8, dim_2), 1, 1], [1, 1, 1, 1], 'VALID'))
        # f3 = flatten(tf.nn.max_pool(h3, [1, min(2, dim_3), 1, 1], [1, 1, 1, 1], 'VALID'))
        # features = tf.concat([f1, f2, f3], axis=1)

        # f1 = flatten(tf.nn.max_pool(h1, [1, dim_1 // 3, 1, 1], [1, 1, 1, 1], 'VALID'))
        # f2 = flatten(tf.nn.max_pool(h2, [1, dim_2 // 2, 1, 1], [1, 1, 1, 1], 'VALID'))
        # f3 = flatten(tf.nn.max_pool(h3, [1, min(2, dim_3), 1, 1], [1, 1, 1, 1], 'VALID'))
        # features = tf.concat([f1, f2, f3], axis=1)

        f1 = flatten(tf.nn.max_pool(h1, [1, dim_1 // 2, 1, 1], [1, 1, 1, 1], 'VALID'))
        f2 = flatten(tf.nn.max_pool(h2, [1, dim_2 // 3, 1, 1], [1, 1, 1, 1], 'VALID'))
        f3 = flatten(tf.nn.max_pool(h3, [1, max(2, dim_3 // 4), 1, 1], [1, 1, 1, 1], 'VALID'))
        features = tf.concat([f1, f2, f3], axis=1)

        # the following good for LSVC and LR, but bad for KNN
        # f1 = flatten(tf.nn.avg_pool(h1, [1, dim_1 // 2, 1, 1], [1, 1, 1, 1], 'VALID'))
        # f2 = flatten(tf.nn.avg_pool(h2, [1, dim_2 // 3, 1, 1], [1, 1, 1, 1], 'VALID'))
        # f3 = flatten(tf.nn.avg_pool(h3, [1, min(2, dim_3), 1, 1], [1, 1, 1, 1], 'VALID'))
        # features = tf.concat([f1, f2, f3], axis=1)

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
