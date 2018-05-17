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
# -*- coding: utf-8 -*-
# File: shapes.py


import tensorflow as tf
from .common import layer_register

__all__ = ['ConcatWith']


@layer_register(use_scope=None)
def ConcatWith(x, tensor, dim):
    """
    A wrapper around ``tf.concat`` to cooperate with :class:`LinearWrap`.

    Args:
        x (tf.Tensor): input
        tensor (list[tf.Tensor]): a tensor or list of tensors to concatenate with x.
            x will be at the beginning
        dim (int): the dimension along which to concatenate

    Returns:
        tf.Tensor: ``tf.concat([x] + tensor, dim)``
    """
    if type(tensor) != list:
        tensor = [tensor]
    return tf.concat([x] + tensor, dim)
