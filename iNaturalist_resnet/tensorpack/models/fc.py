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
# -*- coding: UTF-8 -*-
# File: fc.py


import tensorflow as tf

from .common import layer_register, VariableHolder
from .tflayer import convert_to_tflayer_args, rename_get_variable
from ..tfutils import symbolic_functions as symbf

__all__ = ['FullyConnected']


@layer_register(log_shape=True)
@convert_to_tflayer_args(
    args_names=['units'],
    name_mapping={'out_dim': 'units'})
def FullyConnected(
        inputs,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None):
    """
    A wrapper around `tf.layers.Dense`.
    One difference to maintain backward-compatibility:
    Default weight initializer is variance_scaling_initializer(2.0).

    Variable Names:

    * ``W``: weights of shape [in_dim, out_dim]
    * ``b``: bias
    """

    inputs = symbf.batch_flatten(inputs)
    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Dense(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer)
        ret = layer.apply(inputs, scope=tf.get_variable_scope())
        ret = tf.identity(ret, name='output')

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return ret
