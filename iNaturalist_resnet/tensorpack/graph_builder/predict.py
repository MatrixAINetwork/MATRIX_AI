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
# File: predict.py

import tensorflow as tf

from ..utils import logger
from ..tfutils.tower import TowerContext
from .training import GraphBuilder

__all__ = ['SimplePredictBuilder']


class SimplePredictBuilder(GraphBuilder):
    """
    Single-tower predictor.
    """
    def __init__(self, ns_name='', vs_name='', device=0):
        """
        Args:
            ns_name (str):
            vs_name (str):
            device (int):
        """
        self._ns_name = ns_name
        self._vs_name = vs_name

        device = '/gpu:{}'.format(device) if device >= 0 else '/cpu:0'
        self._device = device

    def build(self, input, tower_fn):
        """
        Args:
            input (InputSource): must have been setup
            tower_fn ( [tf.Tensors] ->): callable that takes input tensors.

        Returns:
            The return value of tower_fn called under the proper context.
        """
        assert input.setup_done()
        logger.info("Building predictor tower '{}' on device {} ...".format(
            self._ns_name, self._device))

        with tf.device(self._device), \
                TowerContext(
                    self._ns_name, is_training=False, vs_name=self._vs_name):
            inputs = input.get_input_tensors()
            assert isinstance(inputs, (list, tuple)), inputs
            return tower_fn(*inputs)
