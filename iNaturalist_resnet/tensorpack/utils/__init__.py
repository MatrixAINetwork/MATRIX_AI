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
#  -*- coding: UTF-8 -*-
#  File: __init__.py


from pkgutil import iter_modules
import os

"""
Common utils.
These utils should be irrelevant to tensorflow.
"""

__all__ = []


# this two functions for back-compat only
def get_nr_gpu():
    from .gpu import get_nr_gpu as gg
    logger.warn(    # noqa
        "get_nr_gpu will not be automatically imported any more! "
        "Please do `from tensorpack.utils.gpu import get_nr_gpu`")
    return gg()


def change_gpu(val):
    from .gpu import change_gpu as cg
    logger.warn(    # noqa
        "change_gpu will not be automatically imported any more! "
        "Please do `from tensorpack.utils.gpu import change_gpu`")
    return cg(val)


def get_rng(obj=None):
    from .utils import get_rng as gr
    logger.warn(    # noqa
        "get_rng will not be automatically imported any more! "
        "Please do `from tensorpack.utils.utils import get_rng`")
    return gr(obj)

# Import no submodules. they are supposed to be explicitly imported by users.
__all__.extend(['logger', 'get_nr_gpu', 'change_gpu', 'get_rng'])
