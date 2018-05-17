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
# File: multigpu.py


import tensorflow as tf
from ..utils import logger
from ..graph_builder.predict import SimplePredictBuilder
from ..graph_builder.model_desc import InputDesc
from ..input_source import PlaceholderInput
from .base import OnlinePredictor

__all__ = ['MultiTowerOfflinePredictor',
           'DataParallelOfflinePredictor']


class MultiTowerOfflinePredictor(OnlinePredictor):
    """ A multi-tower multi-GPU predictor. """

    def __init__(self, config, towers):
        """
        Args:
            config (PredictConfig): the config to use.
            towers: a list of relative GPU id.
        """
        assert len(towers) > 0
        self.graph = config._maybe_create_graph()
        self.predictors = []
        self.return_input = config.return_input
        with self.graph.as_default():
            handles = []

            input = PlaceholderInput()
            input.setup(config.inputs_desc)

            for idx, t in enumerate(towers):
                tower_name = 'tower' + str(t)

                with tf.variable_scope(tf.get_variable_scope(), reuse=idx > 0):
                    builder = SimplePredictBuilder(ns_name=tower_name, device=t)
                    builder.build(input, config.tower_func)
                    handles.append(config.tower_func.towers[-1])

            self.sess = config.session_creator.create_session()
            config.session_init.init(self.sess)

            for h in handles:
                input_tensors = h.get_tensors(config.input_names)
                output_tensors = h.get_tensors(config.output_names)
                self.predictors.append(OnlinePredictor(
                    input_tensors, output_tensors, config.return_input, self.sess))

    def _do_call(self, dp):
        # use the first tower for compatible PredictorBase interface
        return self.predictors[0]._do_call(dp)

    def get_predictor(self, n):
        """
        Returns:
            OnlinePredictor: the nth predictor on the nth tower.
        """
        l = len(self.predictors)
        if n >= l:
            logger.warn("n > #towers, will assign predictor to GPU by round-robin")
        return [self.predictors[k % l] for k in range(n)]

    def get_predictors(self):
        """
        Returns:
            list[OnlinePredictor]: a list of predictor
        """
        return self.predictors


class DataParallelOfflinePredictor(OnlinePredictor):
    """
    A data-parallel predictor.
    Note that it doesn't split/concat inputs/outputs automatically.
    Instead, its inputs are:
    ``[input[0] in tower[0], input[1] in tower[0], ..., input[0] in tower[1], input[1] in tower[1], ...]``
    Similar for the outputs.
    """

    def __init__(self, config, towers):
        """
        Args:
            config (PredictConfig): the config to use.
            towers: a list of relative GPU id.
        """
        self.graph = config._maybe_create_graph()
        with self.graph.as_default():
            input_tensors = []
            output_tensors = []

            for idx, t in enumerate(towers):
                tower_name = 'tower' + str(t)

                inputs_desc = [InputDesc(desc.type, desc.shape, tower_name + '_' + desc.name)
                               for desc in config.inputs_desc]
                input = PlaceholderInput()
                input.setup(inputs_desc)

                with tf.variable_scope(tf.get_variable_scope(), reuse=idx > 0):
                    builder = SimplePredictBuilder(ns_name=tower_name, device=t)
                    builder.build(input, config.tower_func)
                    h = config.tower_func.towers[-1]
                    input_tensors.extend(h.get_tensors(config.input_names))
                    output_tensors.extend(h.get_tensors(config.output_names))

            sess = config.session_creator.create_session()
            config.session_init.init(sess)
            super(DataParallelOfflinePredictor, self).__init__(
                input_tensors, output_tensors, config.return_input, sess)
