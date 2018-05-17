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
# File: __init__.py

STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = False
if STATICA_HACK:
    from .base import *
    from .convert import *
    from .crop import *
    from .deform import *
    from .geometry import *
    from .imgproc import *
    from .meta import *
    from .misc import *
    from .noise import *
    from .paste import *
    from .transform import *


import os
from pkgutil import iter_modules

__all__ = []


def global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    if lst:
        del globals()[name]
        for k in lst:
            if not k.startswith('__'):
                globals()[k] = p.__dict__[k]
                __all__.append(k)


try:
    import cv2  # noqa
except ImportError:
    from ...utils import logger
    logger.warn("Cannot import 'cv2', therefore image augmentation is not available.")
else:
    _CURR_DIR = os.path.dirname(__file__)
    for _, module_name, _ in iter_modules(
            [os.path.dirname(__file__)]):
        srcpath = os.path.join(_CURR_DIR, module_name + '.py')
        if not os.path.isfile(srcpath):
            continue
        if not module_name.startswith('_'):
            global_import(module_name)
