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
from lib.mmd import mix_rbf_mmd2

from utils import *

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
    vis.plot_series(vis_X, os.path.join(conf.dir_samples, "000_real.png"))
    # save variables to log
    save_variables(conf, os.path.join(conf.dir_logs,'variables'))
    f_log_train = open(os.path.join(conf.dir_logs,'log_train.ndjson'), 'w')
    log_fields = [
        'n_epoches',
        'n_updates',
        'n_examples',
        'n_seconds',
        'nnd',
        'mmd',
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

        t = time()
        n_updates = 0
        n_epoches = 0
        n_examples = 0
        g_losses, d_losses, d_losses_fake, d_losses_real = [], [], [], []
        nnds = []
        mmds = []
        mmd_bandwidths = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
        mmd_batchsize = min(conf.nsample, conf.X.shape[0])
        mmd_real_t = tf.placeholder(tf.float32, [mmd_batchsize, conf.img_h], name='mmd_real')
        mmd_sample_t = tf.placeholder(tf.float32, [mmd_batchsize, conf.img_h], name='mmd_sample')
        mmd_loss_t = mix_rbf_mmd2(mmd_real_t, mmd_sample_t, sigmas=mmd_bandwidths)
        # train
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
                print("Epoch: [{}/{}], g_loss = {:.4f}, d_loss = {:.4f}, d_loss_fake = {:.4f}, d_loss_reak = {:.4f}".
                    format(epoch, conf.nepoch, g_loss, d_loss, d_loss_fake, d_loss_real))
            if epoch % conf.freq_log == 0 or epoch == conf.nepoch-1:
                # eval
                gX = gan_sample(sess, gan, conf, conf.nsample)
                gX = gX.reshape(len(gX), -1)
                teX = conf.X.reshape(len(conf.X), -1)
                nnd_ = metrics.nnd_score(gX[:mmd_batchsize], teX[:mmd_batchsize], metric='euclidean')
                nnds.append(nnd_)
                mmd_ = sess.run(mmd_loss_t,
                                feed_dict={mmd_real_t: teX[:mmd_batchsize],
                                           mmd_sample_t: gX[:mmd_batchsize]})
                mmds.append(mmd_)
                log_valus = [n_epoches, n_updates, n_examples, time()-t,
                             nnd_, float(mmd_), float(g_loss), float(d_loss_real), float(d_loss_fake)]
                f_log_train.write(json.dumps(dict(zip(log_fields, log_valus))) + '\n')
                f_log_train.flush()
                # save checkpoint
                gan.save(sess, conf.dir_checkpoint, n_updates, conf.model_name)

            if epoch % conf.freq_plot == 0  or epoch == conf.nepoch - 1:
                samples = gan_sample(sess, gan, conf, vis_nsample)
                samples = samples.reshape([samples.shape[0],-1])
                img_path = os.path.join(conf.dir_samples, "train_{}.png".format(str(epoch+1).zfill(4)))
                txt_path = os.path.join(conf.dir_samples, "train_{}".format(str(epoch+1).zfill(4)))
                vis.plot_series(samples, img_path)
                np.savetxt(txt_path, samples, delimiter=',', newline='\n')

    metrics_dic = {
        'g_loss': np.array(g_losses),
        'd_loss': np.array(d_losses),
        'd_loss_fake': np.array(d_losses_fake),
        'd_loss_real': np.array(d_losses_real),
        'nnd': np.array(nnds),
        'mmd': np.array(mmds)
    }

    metrics_save(metrics_dic, conf)
    metrics_vis(metrics_dic, conf)