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
import tensorflow as tf
import numpy as np

import model
from config import Config
from lib import data_utils
from lib import vis
from lib import load
import lib.classifier.base as clf

from utils import *

tf_conf = tf.ConfigProto()
tf_conf.gpu_options.allow_growth = True


def generate_test_sample(conf):
    gan = model.GAN(conf.z_dim, conf.img_h, conf.img_w, conf.c_dim,
                    conf.g_learning_rate, conf.d_learning_rate,
                    conf.g_beta1, conf.d_beta2,
                    conf.gf_dim, conf.df_dim)
    nsample = 6
    with tf.Session(config=tf_conf) as sess:
        isload, counter = gan.load(sess, conf.dir_checkpoint, conf.model_name)
        if not isload:
            raise Exception("[!] Train a model first, then run test mode")
        for i in range(10):
            print("[*] sample figure {} ..".format(i))
            samples = gan_sample(sess, gan, conf, nsample)
            samples = samples.reshape([samples.shape[0], -1])
            img_path = os.path.join(conf.dir_samples, "test_{}.png".format(str(i + 1).zfill(4)))
            txt_path = os.path.join(conf.dir_samples, "test_{}".format(str(i + 1).zfill(4)))
            vis.plot_series(samples, img_path)
            np.savetxt(txt_path, samples, delimiter=',', newline='\n')


if __name__ == '__main__':
    data_root = '/home/hfl/dataset/timeseries/UCR_TS_Archive_2015'
    fname = 'ArrowHead'
    out_fname = "uncond_dcgan_base_mmd"
    data = load.read_data(data_root, fname)
    trX = np.reshape(data.X_train, data.X_train.shape + (1, 1))
    teX = np.reshape(data.X_test, data.X_test.shape + (1, 1))
    trX = np.vstack([trX, teX])
    height = trX.shape[1]
    width = trX.shape[2]
    c_dim = trX.shape[3]
    conf = Config(trX, fname, out_fname, img_h=height, img_w=width, c_dim=c_dim, state='test')
    generate_test_sample(conf)
