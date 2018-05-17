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


from .tower import get_current_tower_context, TowerContext

STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = False
if STATICA_HACK:
    from .common import *
    from .sessinit import *
    from .argscope import *


# don't want to include everything from .tower
__all__ = ['get_current_tower_context', 'TowerContext']


def _global_import(name):
    p = __import__(name, globals(), None, level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    for k in lst:
        if not k.startswith('__'):
            globals()[k] = p.__dict__[k]
            __all__.append(k)


_TO_IMPORT = frozenset([
    'common',
    'sessinit',
    'argscope',
])

for module_name in _TO_IMPORT:
    _global_import(module_name)

"""
TODO remove this line in the future.
Better to keep submodule names (sesscreate, varmanip, etc) out of __all__,
so that these names will be invisible under `tensorpack.` namespace.

To use these utilities, users are expected to import them explicitly, e.g.:

import tensorpack.tfutils.symbolic_functions as symbf
"""
__all__.extend(['sessinit', 'summary', 'optimizer',
                'sesscreate', 'gradproc', 'varreplace', 'symbolic_functions',
                'distributed', 'tower'])
