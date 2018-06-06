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
from lib import load
import lib.classifier.base as clf
import lib.markdown as md

from utils import *

tf_conf = tf.ConfigProto()
tf_conf.gpu_options.allow_growth = True


def classifier(sess, conf, gan, data):
    trX = data.X_train
    teX = data.X_test
    trY = data.y_train
    teY = data.y_test

    # original data
    acc_lsvc = clf.linearSVC(trX, trY, teX, teY)
    acc_knn = clf.knn(trX, trY, teX, teY)
    acc_lr = clf.logisticRegression(trX, trY, teX, teY)

    # gan features
    trX_gan = np.reshape(trX, trX.shape + (1, 1))
    teX_gan = np.reshape(teX, teX.shape + (1, 1))
    trFeatures = gan_features(trX_gan, sess, conf, gan)
    teFeatures = gan_features(teX_gan, sess, conf, gan)
    acc_lsvc_gan = clf.linearSVC(trFeatures, trY, teFeatures, teY)
    acc_knn_gan = clf.knn(trFeatures, trY, teFeatures, teY)
    acc_lr_gan = clf.logisticRegression(trFeatures, trY, teFeatures, teY)

    print("*"*50 + 'accuracy: ')
    print("acc_lsvc: train-{:.4f}, test-{:.4f}".format(acc_lsvc[0], acc_lsvc[1]))
    print("acc_knn: train-{:.4f}, test-{:.4f}".format(acc_knn[0], acc_knn[1]))
    print("acc_lr: train-{:.4f}, test-{:.4f}".format(acc_lr[0], acc_lr[1]))
    print("acc_feature_lsvc: train-{:.4f}, test-{:.4f}".format(acc_lsvc_gan[0], acc_lsvc_gan[1]))
    print("acc_feature_knn: train-{:.4f}, test-{:.4f}".format(acc_knn_gan[0], acc_knn_gan[1]))
    print("acc_feature_lr: train-{:.4f}, test-{:.4f}".format(acc_lr_gan[0], acc_lr_gan[1]))
    print("*"*50)

    acc_dic = {
        'knn': acc_knn,
        'knn_gan': acc_knn_gan,
        'lsvc': acc_lsvc,
        'lsvc_gan': acc_lsvc_gan,
        'lr': acc_lr,
        'lr_gan': acc_lr_gan
    }
    return acc_dic

def run(conf):
    gan = model.GAN(conf.z_dim, conf.img_h, conf.img_w, conf.c_dim,
                    conf.g_learning_rate, conf.d_learning_rate,
                    conf.g_beta1, conf.d_beta2,
                    conf.gf_dim, conf.df_dim)

    with tf.Session(config=tf_conf) as sess:
        isload, counter = gan.load(sess, conf.dir_checkpoint, conf.model_name)
        if not isload:
            raise Exception("[!] Train a model first, then run test mode")

        acc_dic = classifier(sess, conf, gan, data)
        return acc_dic


if __name__ == '__main__':
    data_root = '/home/hfl/dataset/timeseries/UCR_TS_Archive_2015'
    out_fname = "uncond_dcgan_base_mmd"

    fname = 'ArrowHead'
    # load data and initialize configure
    print("**************{}***************".format(fname))
    data = load.read_data(data_root, fname)
    trX = np.reshape(data.X_train, data.X_train.shape + (1, 1))
    teX = np.reshape(data.X_test, data.X_test.shape + (1, 1))
    X = np.vstack([trX, teX])
    # run classifier
    conf = Config(X, fname, out_fname, img_h=X.shape[1], img_w=X.shape[2], c_dim=X.shape[3], state='test')
    acc_dic = run(conf)

