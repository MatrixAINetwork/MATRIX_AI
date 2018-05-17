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
    from .batch_norm import *
    from .common import *
    from .conv2d import *
    from .fc import *
    from .image_sample import *
    from .layer_norm import *
    from .linearwrap import *
    from .nonlin import *
    from .pool import *
    from .regularize import *


from pkgutil import iter_modules
import os
import os.path
# this line is necessary for _TFModuleFunc to work
import tensorflow as tf  # noqa: F401

__all__ = []


def _global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    del globals()[name]
    for k in lst:
        if not k.startswith('__'):
            globals()[k] = p.__dict__[k]
            __all__.append(k)


_CURR_DIR = os.path.dirname(__file__)
_SKIP = ['utils', 'registry', 'tflayer']
for _, module_name, _ in iter_modules(
        [_CURR_DIR]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if module_name.startswith('_'):
        continue
    if module_name not in _SKIP:
        _global_import(module_name)
