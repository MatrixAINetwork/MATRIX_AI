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
# File: globvars.py


import six
import argparse
from . import logger

__all__ = ['globalns', 'GlobalNS']

if six.PY2:
    class NS:
        pass
else:
    import types
    NS = types.SimpleNamespace


# TODO make it singleton

class GlobalNS(NS):
    """
    The class of the globalns instance.
    """
    def use_argument(self, args):
        """
        Add the content of :class:`argparse.Namespace` to this ns.

        Args:
            args (argparse.Namespace): arguments
        """
        assert isinstance(args, argparse.Namespace), type(args)
        for k, v in six.iteritems(vars(args)):
            if hasattr(self, k):
                logger.warn("Attribute {} in globalns will be overwritten!")
            setattr(self, k, v)


globalns = GlobalNS()
"""
A namespace to store global variables.

Examples:

.. code-block:: none

    import tensorpack.utils.globalns as G

    G.depth = 18
    G.batch_size = 1
    G.use_argument(parser.parse_args())
"""
