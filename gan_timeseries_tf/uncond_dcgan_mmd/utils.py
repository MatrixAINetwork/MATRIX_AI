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

import numpy as np
from six.moves import xrange
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from lib import utils
from lib import rng

def metrics_save(metrics_dic, conf):
    f_metrics = open(os.path.join(conf.dir_logs, 'metrics'), 'w')
    for key, y in metrics_dic.items():
        values = ''.join('{},'.format(e) for e in metrics_dic[key])[:-1]
        f_metrics.write('{}:{}\n'.format(key, values))
    f_metrics.close()

def metrics_vis(metrics_dic, conf):
    plt.figure()
    plt.plot(metrics_dic['g_loss'], label='g_loss')
    plt.plot(metrics_dic['d_loss'], label='d_loss')
    plt.legend(loc='best')
    plt.savefig(os.path.join(conf.dir_logs, 'loss.png'))
    plt.close()

    plt.figure()
    plt.plot(metrics_dic['g_loss'], label='g_loss')
    plt.plot(metrics_dic['d_loss'], label='d_loss')
    plt.plot(metrics_dic['d_loss_fake'], label='d_loss_fake')
    plt.plot(metrics_dic['d_loss_real'], label='d_loss_real')
    plt.legend(loc='best')
    plt.savefig(os.path.join(conf.dir_logs, 'loss_all.png'))
    plt.close()

    plt.figure()
    plt.plot(xrange(0, metrics_dic['mmd'].shape[0] * conf.freq_log, conf.freq_log),
             metrics_dic['mmd'], '^-')
    plt.savefig(os.path.join(conf.dir_logs, 'mmd.png'))
    plt.close()

    plt.figure()
    plt.plot(xrange(0, metrics_dic['nnd'].shape[0] * conf.freq_log, conf.freq_log),
             metrics_dic['nnd'], '^-')
    plt.savefig(os.path.join(conf.dir_logs, 'nnd.png'))
    plt.close()


def sample_z(size):
    return rng.np_rng.uniform(-1, 1, size)


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