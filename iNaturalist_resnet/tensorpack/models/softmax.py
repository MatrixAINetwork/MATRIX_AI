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
#!/usr/bin/env python
# -*- coding: utf-8 -*- File: softmax.py


import tensorflow as tf
from .common import layer_register
from ..utils.develop import log_deprecated

__all__ = ['SoftMax']


@layer_register(use_scope=None)
def SoftMax(x, use_temperature=False, temperature_init=1.0):
    """
    A SoftMax layer (w/o linear projection) with optional temperature, as
    defined in the paper `Distilling the Knowledge in a Neural Network
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        x (tf.Tensor): input of any dimension. Softmax will be performed on
            the last dimension.
        use_temperature (bool): use a learnable temperature or not.
        temperature_init (float): initial value of the temperature.

    Returns:
        tf.Tensor: a tensor of the same shape named ``output``.

    Variable Names:

    * ``invtemp``: 1.0/temperature.
    """
    log_deprecated("models.SoftMax", "Please implement it by yourself!", "2018-05-01")
    if use_temperature:
        t = tf.get_variable('invtemp', [],
                            initializer=tf.constant_initializer(1.0 / float(temperature_init)))
        x = x * t
    return tf.nn.softmax(x, name='output')
