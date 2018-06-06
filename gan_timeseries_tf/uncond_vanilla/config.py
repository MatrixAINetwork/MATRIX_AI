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

import os
import shutil

class Config(object):
    def __init__(self, X, data_name, out_fname,
                 img_h, img_w, c_dim, state='train'):
        # data parameter
        self.X = X
        self.img_h = img_h
        self.img_w = img_w
        self.c_dim = c_dim
        self.z_dim = int(X.shape[1] * 0.7)
        #self.z_dim = 100

        # model parameter
        self.gf_dim = 128  # the number of hidden units in G
        self.df_dim = 128  # the number of hidden units in D

        # training parameter
        self.nbatch = min(20, len(self.X))
        self.g_learning_rate = 0.0005
        self.d_learning_rate = 0.001
        self.g_beta1 = 0.5
        self.d_beta2 = 0.5
        self.nepoch = 500
        self.k = 1

        # log parameter
        self.data_name = data_name
        self.file_name = out_fname
        self.model_name = "{}_{}".format(self.file_name, self.data_name)
        self.freq_print = 1
        self.freq_plot = 20
        self.freq_log = 50
        self.nsample = 100000

        self.dir_root = '/home/hfl/dataset/output/gan_timeseries_tf/{}/{}' \
                        ''.format(self.file_name, self.data_name)
        self.dir_logs = os.path.join(self.dir_root, 'logs')
        self.dir_samples = os.path.join(self.dir_root, 'samples')
        self.dir_checkpoint = os.path.join(self.dir_root, 'checkpoint')

        # construct log file
        if state == 'train':
            if os.path.exists(self.dir_root):
                shutil.rmtree(self.dir_root)
            os.makedirs(self.dir_root)
            os.makedirs(self.dir_logs)
            os.makedirs(self.dir_samples)
            os.makedirs(self.dir_checkpoint)
