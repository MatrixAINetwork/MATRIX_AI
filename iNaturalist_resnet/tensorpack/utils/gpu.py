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
# File: gpu.py


import os
from .utils import change_env
from . import logger
from .nvml import NVMLContext
from .concurrency import subproc_call

__all__ = ['change_gpu', 'get_nr_gpu']


def change_gpu(val):
    """
    Returns:
        a context where ``CUDA_VISIBLE_DEVICES=val``.
    """
    val = str(val)
    if val == '-1':
        val = ''
    return change_env('CUDA_VISIBLE_DEVICES', val)


def get_nr_gpu():
    """
    Returns:
        int: #available GPUs in CUDA_VISIBLE_DEVICES, or in the system.
    """
    env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if env is not None:
        return len(env.split(','))
    output, code = subproc_call("nvidia-smi -L", timeout=5)
    if code == 0:
        output = output.decode('utf-8')
        return len(output.strip().split('\n'))
    else:
        try:
            # Use NVML to query device properties
            with NVMLContext() as ctx:
                return ctx.num_devices()
        except Exception:
            # Fallback
            # Note this will initialize all GPUs and therefore has side effect
            # https://github.com/tensorflow/tensorflow/issues/8136
            logger.info("Loading local devices by TensorFlow ...")
            from tensorflow.python.client import device_lib
            local_device_protos = device_lib.list_local_devices()
            return len([x.name for x in local_device_protos if x.device_type == 'GPU'])
