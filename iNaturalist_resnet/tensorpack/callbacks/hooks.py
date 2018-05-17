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
# File: hooks.py


""" Compatible layers between tf.train.SessionRunHook and Callback"""

import tensorflow as tf
from .base import Callback


__all__ = ['CallbackToHook', 'HookToCallback']


class CallbackToHook(tf.train.SessionRunHook):
    """ This is only for internal implementation of
        before_run/after_run callbacks.
        You shouldn't need to use this.
    """

    _chief_only = False

    def __init__(self, cb):
        self._cb = cb

    def before_run(self, ctx):
        return self._cb.before_run(ctx)

    def after_run(self, ctx, vals):
        self._cb.after_run(ctx, vals)


class HookToCallback(Callback):
    """
    Make a ``tf.train.SessionRunHook`` into a callback.
    Note that the `coord` argument in `after_create_session` will be None.
    """

    _chief_only = False

    def __init__(self, hook):
        """
        Args:
            hook (tf.train.SessionRunHook):
        """
        self._hook = hook

    def _setup_graph(self):
        with tf.name_scope(None):   # jump out of the name scope
            self._hook.begin()

    def _before_train(self):
        sess = tf.get_default_session()
        # coord is set to None when converting
        self._hook.after_create_session(sess, None)

    def _before_run(self, ctx):
        return self._hook.before_run(ctx)

    def _after_run(self, ctx, run_values):
        self._hook.after_run(ctx, run_values)

    def _after_train(self):
        self._hook.end(self.trainer.sess)
