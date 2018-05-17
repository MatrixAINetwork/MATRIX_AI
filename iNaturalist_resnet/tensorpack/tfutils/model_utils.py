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
"""
# -*- coding: UTF-8 -*-
# File: model_utils.py
# Author: tensorpack contributors

import tensorflow as tf
from termcolor import colored
from tabulate import tabulate

from ..utils import logger

__all__ = []


def describe_trainable_vars():
    """
    Print a description of the current model parameters.
    Skip variables starting with "tower", as they are just duplicates built by data-parallel logic.
    """
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    if len(train_vars) == 0:
        logger.warn("No trainable variables in the graph!")
        return
    total = 0
    total_bytes = 0
    data = []
    for v in train_vars:
        if v.name.startswith('tower'):
            continue
        shape = v.get_shape()
        ele = shape.num_elements()
        total += ele
        total_bytes += ele * v.dtype.size
        data.append([v.name, shape.as_list(), ele, v.device, v.dtype.base_dtype.name])
    headers = ['name', 'shape', 'dim', 'device', 'dtype']

    dtypes = set([x[4] for x in data])
    if len(dtypes) == 1:
        for x in data:
            del x[4]
        del headers[4]

    devices = set([x[3] for x in data])
    if len(devices) == 1:
        # don't log the device if all vars on the same device
        for x in data:
            del x[3]
        del headers[3]

    table = tabulate(data, headers=headers)

    size_mb = total_bytes / 1024.0**2
    summary_msg = colored(
        "\nTotal #vars={}, #params={}, size={:.02f}MB".format(
            len(data), total, size_mb), 'cyan')
    logger.info(colored("Trainable Variables: \n", 'cyan') + table + summary_msg)


def get_shape_str(tensors):
    """
    Internally used by layer registry, to print shapes of inputs/outputs of layers.

    Args:
        tensors (list or tf.Tensor): a tensor or a list of tensors
    Returns:
        str: a string to describe the shape
    """
    if isinstance(tensors, (list, tuple)):
        for v in tensors:
            assert isinstance(v, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(v))
        shape_str = ",".join(
            map(lambda x: str(x.get_shape().as_list()), tensors))
    else:
        assert isinstance(tensors, (tf.Tensor, tf.Variable)), "Not a tensor: {}".format(type(tensors))
        shape_str = str(tensors.get_shape().as_list())
    return shape_str
