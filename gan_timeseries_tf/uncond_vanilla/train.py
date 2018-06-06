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

import sys
sys.path.append('..')

import os
import numpy as np
import tensorflow as tf

from six.moves import xrange
import json
from time import time

import model
from lib import data_utils
from lib import vis
from lib import metrics
from lib import utils
from lib import rng

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
tf_conf = tf.ConfigProto()
tf_conf.gpu_options.allow_growth = True

def train(conf):
    # set up object
    gan = model.GAN(conf.z_dim, conf.img_h, conf.img_w, conf.c_dim,
                    conf.g_learning_rate, conf.d_learning_rate,
                    conf.g_beta1, conf.d_beta2,
                    conf.gf_dim, conf.df_dim)
    sample_x = data_utils.DataSet(conf.X)

    # log ground truth
    vis_nsample = min(6, conf.nbatch)
    vis_X = conf.X[:vis_nsample]
    vis_X = vis_X.reshape([vis_X.shape[0], -1])
    vis.plot_series(
        vis_X, os.path.join(conf.dir_samples, "000_real.png"))
    # save variables to log
    save_variables(conf, os.path.join(conf.dir_logs,'variables_{}'.format(conf.model_name)))
    f_log_train = open(os.path.join(conf.dir_logs,'log_train_{}.ndjson'.format(conf.model_name)), 'w')
    log_fields = [
        'n_epoches',
        'n_updates',
        'n_examples',
        'n_seconds',
        '1k_va_nnd',
        '10k_va_nnd',
        '100k_va_nnd',
        'g_loss',
        'd_loss_real',
        'd_loss_fake'
    ]

    # set up tf session and train model
    with tf.Session(config=tf_conf) as sess:
        # initialize
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # train
        n_updates = 0
        n_epoches = 0
        n_examples = 0
        g_losses, d_losses, d_losses_fake, d_losses_real = [], [], [], []
        nnd_1k, nnd_10k, nnd_100k = [], [], []
        t = time()
        for epoch in xrange(conf.nepoch):
            g_loss, d_loss, d_loss_fake, d_loss_real = np.zeros(4)
            for i in xrange(sample_x.num_examples // conf.nbatch):
                x = sample_x.next_batch(conf.nbatch)
                z = sample_z([conf.nbatch, conf.z_dim])

                _ = sess.run(gan.d_opt, feed_dict={gan.x: x, gan.z: z})
                _ = sess.run(gan.g_opt, feed_dict={gan.z: z})
                _ = sess.run(gan.g_opt, feed_dict={gan.z: z})

                d_loss, d_loss_real, d_loss_fake, g_loss = sess.run(
                    [gan.d_loss, gan.d_loss_real, gan.d_loss_fake, gan.g_loss],
                    feed_dict={gan.x: x, gan.z: z})
                n_updates += 1
                n_examples += len(x)
            n_epoches += 1
            g_losses.append(g_loss)
            d_losses.append(d_loss)
            d_losses_fake.append(d_loss_fake)
            d_losses_real.append(d_loss_real)

            # log
            if epoch % conf.freq_print == 0:
                print("Epoch: [{}/{}], g_loss = {:.4f}, d_loss = {:.4f}, "
                      "d_loss_fake = {:.4f}, d_loss_reak = {:.4f}".format(
                    epoch, conf.nepoch,
                    g_loss, d_loss, d_loss_fake, d_loss_real))
            if epoch % conf.freq_log == 0:
                # eval
                gX = gan_sample(sess, gan, conf, conf.nsample)
                gX = gX.reshape(len(gX), -1)
                teX = conf.X.reshape(len(conf.X), -1)
                # teX = conf.teX.reshape(len(conf.teX), -1)
                va_nnd_1k = metrics.nnd_score(gX[:1000], teX, metric='euclidean')
                va_nnd_10k = metrics.nnd_score(gX[:10000], teX, metric='euclidean')
                va_nnd_100k = metrics.nnd_score(gX[:100000], teX, metric='euclidean')
                nnd_1k.append(va_nnd_1k)
                nnd_10k.append(va_nnd_10k)
                nnd_100k.append(va_nnd_100k)

                log_valus = [n_epoches, n_updates, n_examples, time()-t,
                             va_nnd_1k, va_nnd_10k, va_nnd_100k,
                             float(g_loss), float(d_loss_real), float(d_loss_fake)]
                f_log_train.write(
                    json.dumps(dict(zip(log_fields, log_valus))) + '\n')
                f_log_train.flush()
                # save checkpoint
                gan.save(sess, conf.dir_checkpoint, n_updates, conf.model_name)

            if epoch % conf.freq_plot == 0:
                samples = gan_sample(sess, gan, conf, vis_nsample)
                samples = samples.reshape([samples.shape[0],-1])
                img_path = os.path.join(
                    conf.dir_samples,"train_{}.png".format(str(epoch+1).zfill(4)))
                vis.plot_series(samples, img_path)

        # plot loss
        losses = {'g_loss': np.array(g_losses),
                  'd_loss': np.array(d_losses),
                  'd_loss_fake': np.array(d_losses_fake),
                  'd_loss_real': np.array(d_losses_real)}
        vis.plot_dic(losses, title='{}_loss'.format(conf.data_name),
                     save_path=os.path.join(conf.dir_logs, 'loss_{}.png'.format(conf.model_name)))
        nnd = {'nnd_1k': np.array(nnd_1k),
               'nnd_10k': np.array(nnd_10k),
               'nnd_100k': np.array(nnd_100k)}
        vis.plot_dic(nnd, title='{}_nnd'.format(conf.data_name),
                     save_path=os.path.join(conf.dir_logs, 'nnd_{}.png'.format(conf.model_name)))


def gan_features(X, sess, conf, gan):
    """extract feature from the discriminator of GAN"""
    features = []
    nsamples = len(X)
    n_map = 0
    for i in xrange(nsamples // conf.nbatch):
        x_batch = X[i * conf.nbatch:(i + 1) * conf.nbatch]
        f_batch = sess.run(gan.d_features, feed_dict={gan.x: x_batch})
        features.append(f_batch)
        n_map += len(f_batch)
    n_left = nsamples - n_map
    if n_left > 0:
        x_left = X[-n_left:]
        f_left = sess.run(gan.d_features, feed_dict={gan.x: x_left})
        features.append(f_left)
    features = np.concatenate(features, axis=0)
    return features

def sample_z(size):
    return rng.np_rng.uniform(-1, 1, size)

def gan_sample(sess, gan, conf, nsample):
    """generate samples with generator of GAN"""
    samples = []
    n_gen = 0
    for i in xrange(nsample // conf.nbatch):
        z = sample_z([conf.nbatch, conf.z_dim])
        samples_batch = sess.run(gan.sampler, feed_dict={gan.z: z})
        samples.append(samples_batch)
        n_gen += len(samples_batch)
    n_left = nsample - n_gen
    if n_left > 0:
        z = sample_z([n_left, conf.z_dim])
        samples_left = sess.run(gan.sampler, feed_dict={gan.z: z})
        samples.append(samples_left)
    return np.concatenate(samples, axis=0)

def save_variables(conf, save_path):
    """save the configure and model variables"""
    vars_conf = utils.analyze_object_variables(conf, print_info=True)
    vars_tensor = utils.analyze_tensor_variables(tf.trainable_variables(), print_info=True)
    vars_list = list()
    vars_list.append("=" * 80)
    vars_list.append("variables of configure")
    vars_list.append("-" * 80)
    vars_list.extend(vars_conf)
    vars_list.append("=" * 80)
    vars_list.append("=" * 80 + "\n")
    vars_list.append("variables of tensor")
    vars_list.append("-" * 80)
    vars_list.extend(vars_tensor)

    utils.save_variables_to_file(save_path, vars_list)