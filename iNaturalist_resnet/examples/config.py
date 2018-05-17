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
#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, math
import os.path as osp
from easydict import EasyDict as edict
from datetime import datetime


gpu_ids = "7"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

__C = edict()
cfg = __C
__C.TRAIN = edict()

__C.user = 'steve'

###################generate some dirs, e.g. dump dir, weights dir#############
__C.DATA_DIR = '/home/steve/DataSet/iNaturalist2018'
__C.gt_sets = ['train2018.json', 'val2018.json', 'test2018.json']
__C.ground_truth_dir = osp.abspath(osp.join(__C.DATA_DIR, 'ground_truth'))
__C.test2018_dir = osp.abspath(osp.join(__C.DATA_DIR, 'test2018'))
__C.train_val2018_dir = osp.abspath(osp.join(__C.DATA_DIR, 'train_val2018'))

iter = 239190
__C.weights_dir = './train_log/iNaturalist-resnet-d152'
__C.weights_filename = osp.join(cfg.weights_dir, 'model-{:d}'.format(iter))
__C.meta_filename = osp.join(cfg.weights_dir, 'graph-0506-104408.meta')

#################classes and number of every class######################
__C.superclasses = ['Actinopterygii','Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Bacteria', 'Chromista',
                    'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Protozoa', 'Reptilia']
__C.num_classes = 8142
__C.num_trainset = 437513
__C.num_trainset = 149394
__C.batch_size = 64
__C.img_size = 224


if __name__ == '__main__':
    pass
