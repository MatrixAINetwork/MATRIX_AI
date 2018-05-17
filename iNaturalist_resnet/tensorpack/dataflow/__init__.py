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

STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = False
if STATICA_HACK:
    from .base import *
    from .common import *
    from .format import *
    from .image import *
    from .parallel_map import *
    from .parallel import *
    from .raw import *
    from .remote import *
    from . import imgaug
    from . import dataset
    from . import dftools


from pkgutil import iter_modules
import os
import os.path
from ..utils.develop import LazyLoader

__all__ = []


def _global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    if lst:
        del globals()[name]
        for k in lst:
            if not k.startswith('__'):
                globals()[k] = p.__dict__[k]
                __all__.append(k)


__SKIP = set(['dftools', 'dataset', 'imgaug'])
_CURR_DIR = os.path.dirname(__file__)
for _, module_name, __ in iter_modules(
        [os.path.dirname(__file__)]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if not module_name.startswith('_') and \
            module_name not in __SKIP:
        _global_import(module_name)


globals()['dataset'] = LazyLoader('dataset', globals(), 'tensorpack.dataflow.dataset')
globals()['imgaug'] = LazyLoader('imgaug', globals(), 'tensorpack.dataflow.imgaug')

del LazyLoader

__all__.extend(['imgaug', 'dftools', 'dataset'])
