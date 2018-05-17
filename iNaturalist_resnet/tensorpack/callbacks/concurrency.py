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
# File: concurrency.py


import multiprocessing as mp
from .base import Callback
from ..utils.concurrency import start_proc_mask_signal, StoppableThread
from ..utils import logger

__all__ = ['StartProcOrThread']


class StartProcOrThread(Callback):
    """
    Start some threads or processes before training.
    """

    _chief_only = False

    def __init__(self, startable, stop_at_last=True):
        """
        Args:
            startable (list): list of processes or threads which have ``start()`` method.
                Can also be a single instance of process of thread.
            stop_at_last (bool): whether to stop the processes or threads
                after training. It will use :meth:`Process.terminate()` or
                :meth:`StoppableThread.stop()`, but will do nothing on normal
                `threading.Thread` or other startable objects.
        """
        if not isinstance(startable, list):
            startable = [startable]
        self._procs_threads = startable
        self._stop_at_last = stop_at_last

    def _before_train(self):
        logger.info("Starting " +
                    ', '.join([k.name for k in self._procs_threads]) + ' ...')
        # avoid sigint get handled by other processes
        start_proc_mask_signal(self._procs_threads)

    def _after_train(self):
        if not self._stop_at_last:
            return
        for k in self._procs_threads:
            if not k.is_alive():
                continue
            if isinstance(k, mp.Process):
                logger.info("Stopping {} ...".format(k.name))
                k.terminate()
                k.join(5.0)
                if k.is_alive():
                    logger.error("Cannot join process {}.".format(k.name))
            elif isinstance(k, StoppableThread):
                logger.info("Stopping {} ...".format(k.name))
                k.stop()
                k.join(5.0)
                if k.is_alive():
                    logger.error("Cannot join thread {}.".format(k.name))
