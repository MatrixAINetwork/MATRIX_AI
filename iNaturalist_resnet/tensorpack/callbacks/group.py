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
# File: group.py


import tensorflow as tf
from contextlib import contextmanager
import time
import traceback

from .base import Callback
from .hooks import CallbackToHook
from ..utils import logger
from ..utils.utils import humanize_time_delta

__all__ = ['Callbacks']


class CallbackTimeLogger(object):
    def __init__(self):
        self.times = []
        self.tot = 0

    def add(self, name, time):
        self.tot += time
        self.times.append((name, time))

    @contextmanager
    def timed_callback(self, name):
        s = time.time()
        yield
        self.add(name, time.time() - s)

    def log(self):

        """ log the time of some heavy callbacks """
        if self.tot < 3:
            return
        msgs = []
        for name, t in self.times:
            if t / self.tot > 0.3 and t > 1:
                msgs.append(name + ": " + humanize_time_delta(t))
        logger.info(
            "Callbacks took {:.3f} sec in total. {}".format(
                self.tot, '; '.join(msgs)))


class Callbacks(Callback):
    """
    A container to hold all callbacks, and trigger them iteratively.
    Note that it does nothing to before_run/after_run.
    """

    def __init__(self, cbs):
        """
        Args:
            cbs(list): a list of :class:`Callback` instances.
        """
        # check type
        for cb in cbs:
            assert isinstance(cb, Callback), cb.__class__
        self.cbs = cbs

    def _setup_graph(self):
        with tf.name_scope(None):   # clear the name scope
            for cb in self.cbs:
                cb.setup_graph(self.trainer)

    def _before_train(self):
        for cb in self.cbs:
            cb.before_train()

    def _after_train(self):
        for cb in self.cbs:
            # make sure callbacks are properly finalized
            try:
                cb.after_train()
            except Exception:
                traceback.print_exc()

    def get_hooks(self):
        return [CallbackToHook(cb) for cb in self.cbs]

    def trigger_step(self):
        for cb in self.cbs:
            cb.trigger_step()

    def _trigger_epoch(self):
        tm = CallbackTimeLogger()

        for cb in self.cbs:
            display_name = str(cb)
            with tm.timed_callback(display_name):
                cb.trigger_epoch()
        tm.log()

    def _before_epoch(self):
        for cb in self.cbs:
            cb.before_epoch()

    def _after_epoch(self):
        for cb in self.cbs:
            cb.after_epoch()
