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
# File: sesscreate.py


import tensorflow as tf
from .common import get_default_sess_config
from ..utils import logger

__all__ = ['NewSessionCreator', 'ReuseSessionCreator', 'SessionCreatorAdapter']

"""
A SessionCreator should:
    (optionally) finalize the graph
    create the session
    initialize all variables
    return a session that is ready to use
"""


class NewSessionCreator(tf.train.ChiefSessionCreator):
    def __init__(self, target='', graph=None, config=None):
        """
        Args:
            target, graph, config: same as :meth:`Session.__init__()`.
            config: defaults to :func:`tfutils.get_default_sess_config()`
        """
        assert graph is None

        if config is None:
            # distributd trainer doesn't support user-provided config
            # we set this attribute so that they can check
            self.user_provided_config = False
            config = get_default_sess_config()
        else:
            self.user_provided_config = True
            logger.warn(
                "Some options in custom session config may not work due to TF \
bugs. See https://github.com/ppwwyyxx/tensorpack/issues/497 for workarounds.")

        self.config = config
        super(NewSessionCreator, self).__init__(master=target, config=config)


class ReuseSessionCreator(tf.train.SessionCreator):
    def __init__(self, sess):
        """
        Args:
            sess (tf.Session): the session to reuse
        """
        self.sess = sess

    def create_session(self):
        return self.sess


class SessionCreatorAdapter(tf.train.SessionCreator):
    def __init__(self, session_creator, func):
        """
        Args:
            session_creator (tf.train.SessionCreator): a session creator
            func (tf.Session -> tf.Session): takes a session created by
            ``session_creator``, and return a new session to be returned by ``self.create_session``
        """
        self._creator = session_creator
        self._func = func

    def create_session(self):
        sess = self._creator.create_session()
        return self._func(sess)
