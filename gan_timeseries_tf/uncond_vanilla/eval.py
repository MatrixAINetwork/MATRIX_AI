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

import model
from config import Config
from lib import data_utils
from lib import vis
import train

tf_conf = tf.ConfigProto()
tf_conf.gpu_options.allow_growth = True


def test(conf):
    gan = model.GAN(conf.z_dim, conf.x_dim,
                    conf.g_hidden_units, conf.d_hidden_units,
                    conf.g_learning_rate, conf.d_learning_rate)

    with tf.Session(config=tf_conf) as sess:
        isload, counter = gan.load(sess, conf.dir_checkpoint, conf.model_name)
        if not isload:
            raise Exception("[!] Train a model first, then run test mode")
        for i in range(10):
            print("[*] sample figure {} ..".format(i))
            nsample = conf.map_height * conf.map_width
            samples = train.gan_sample(sess, gan, conf, nsample)
            samples = samples.reshape([samples.shape[0],
                                       conf.image_height, conf.image_width])
            img_path = os.path.join(conf.dir_samples,
                                    "test_{}.png".format(str(i+1).zfill(4)))
            vis.grid_vis(samples, conf.map_height, conf.map_width,
                         conf.c_dim==1, img_path)


if __name__ == '__main__':
    data_dir = "/home/hfl/dataset/images/mnist_unpack"
    data = data_utils.load_mnist(data_dir)
    X = data.X_train
    conf = Config(X, 'mnist', img_h=28, img_w=28, c_dim=1,
                  state='test', y=data.y_train, y_dim=10)
    test(conf)
