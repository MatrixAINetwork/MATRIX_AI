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
# -*- coding: utf-8 -*-
# File: __init__.py


import os as _os

from tensorpack.libinfo import __version__, _HAS_TF

from tensorpack.utils import *
from tensorpack.dataflow import *

# dataflow can be used alone without installing tensorflow
# TODO maybe separate dataflow to a new project if it's good enough

STATICA_HACK = True
globals()['kcah_acitats'[::-1].upper()] = _HAS_TF
if STATICA_HACK:
    from tensorpack.models import *

    from tensorpack.callbacks import *
    from tensorpack.tfutils import *

    # Default to v2
    if _os.environ.get('TENSORPACK_TRAIN_API', 'v2') == 'v2':
        from tensorpack.train import *
    else:
        from tensorpack.trainv1 import *
    from tensorpack.graph_builder import InputDesc, ModelDesc, ModelDescBase
    from tensorpack.input_source import *
    from tensorpack.predict import *
