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


from config import Config

from utils import *
from lib import vis
from lib import load

tf_conf = tf.ConfigProto()
tf_conf.gpu_options.allow_growth = True


if __name__ == '__main__':
    data_root = '/home/hfl/dataset/timeseries/UCR_TS_Archive_2015'
    out_fname = "uncond_dcgan_base_mmd"
    root_project = '/home/hfl/dataset/output/gan_timeseries_tf/{}'.format(out_fname)

    fname_data_list = [f for f in os.listdir(data_root) if os.path.isfile(f) is False]
    fname_data_list = np.sort(fname_data_list)
    for fname in fname_data_list:
        #fname = 'ArrowHead'
        # load data and initialize configure
        print("**************{}***************".format(fname))
        data = load.read_data(data_root, fname)
        trX = np.reshape(data.X_train, data.X_train.shape + (1, 1))
        teX = np.reshape(data.X_test, data.X_test.shape + (1, 1))
        X = np.vstack([trX, teX])

        conf = Config(X, fname, out_fname, img_h=X.shape[1], img_w=X.shape[2], c_dim=X.shape[3], state='test')
        noise = sample_z([6, conf.z_dim])
        img_path = '{}/000-result/{}-train_0000.png'.format(root_project, fname)
        vis.plot_series(noise, img_path)


