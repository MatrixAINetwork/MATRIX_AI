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
import shutil

import lib.markdown as md


if __name__ == '__main__':
    root_data = '/home/hfl/dataset/timeseries/UCR_TS_Archive_2015'
    fname_model = 'uncond_dcgan_base_mmd'
    root_project = '/home/hfl/dataset/output/gan_timeseries_tf/{}'.format(fname_model)
    fname_out = '000-result'
    dir_out = '{}/{}'.format(root_project, fname_out)
    if os.path.exists(dir_out):
        shutil.rmtree(dir_out)
    os.makedirs(dir_out)

    fname_data_list = [f for f in os.listdir(root_data) if os.path.isfile(f) is False]
    fname_data_list = np.sort(fname_data_list)
    if len(fname_data_list) == 0:
        print("there are no any file can be operated!!!!!")
        exit(1)

    # copy image result
    imgs_sample = ['000_real.png', 'train_0001.png', 'train_0200.png']
    imgs_metrics = ['loss_all.png', 'loss.png', 'mmd.png', 'nnd.png']
    for fname in fname_data_list:
        for img in imgs_sample:
            shutil.copy('{}/{}/{}/{}'.format(root_project, fname, 'samples', img),
                        '{}/{}-{}'.format(dir_out, fname, img))
        for img in imgs_metrics:
            shutil.copy('{}/{}/{}/{}'.format(root_project, fname, 'logs', img),
                        '{}/{}-{}'.format(dir_out, fname, img))

    ### copy metrics
    metrics_dic = {}
    file_mkdown = open('{}/000-metrics.md'.format(dir_out), 'w')
    # get metrics key
    with open('{}/{}/logs/metrics'.format(root_project, fname_data_list[0]), 'r') as f:
        lines = f.readlines()
        keys = ['dataset']
        for line in lines:
            key, _ = line.split(':')
            keys.append(key)
        tb_head_str = md.table_row(keys)
        file_mkdown.write(tb_head_str)
        tb_head_line_str = md.table_head_line(len(tb_head_str))
        file_mkdown.write(tb_head_line_str)
    # get metrics values
    for fname in fname_data_list:
        fname_metrics = '{}/{}/logs/metrics'.format(root_project, fname)
        with open(fname_metrics, 'r') as f:
            lines = f.readlines()
            tb_row_elem = [fname]
            for line in lines:
                key, value = line.strip().split(':')
                values = value.split(',')
                epoch1 = '{:.4f}'.format(float(values[0]))
                epochn = '{:.4f}'.format(float(values[-1]))
                tb_row_elem.append('{},{}'.format(epoch1, epochn))
            tb_row_elem_str = md.table_row(tb_row_elem)
            file_mkdown.write(tb_row_elem_str)

    file_mkdown.close()

