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
# File: serialize.py

from .develop import create_dummy_func

__all__ = ['loads', 'dumps']


def dumps_msgpack(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return msgpack.dumps(obj, use_bin_type=True)


def loads_msgpack(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return msgpack.loads(buf, raw=False)


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


try:
    # fixed in pyarrow 0.9: https://github.com/apache/arrow/pull/1223#issuecomment-359895666
    import pyarrow as pa
except ImportError:
    pa = None
    dumps_pyarrow = create_dummy_func('dumps_pyarrow', ['pyarrow'])  # noqa
    loads_pyarrow = create_dummy_func('loads_pyarrow', ['pyarrow'])  # noqa

try:
    import msgpack
    import msgpack_numpy
    msgpack_numpy.patch()
except ImportError:
    assert pa is not None, "pyarrow is a dependency of tensorpack!"
    loads_msgpack = create_dummy_func(  # noqa
        'loads_msgpack', ['msgpack', 'msgpack_numpy'])
    dumps_msgpack = create_dummy_func(  # noqa
        'dumps_msgpack', ['msgpack', 'msgpack_numpy'])

if pa is None:
    loads = loads_msgpack
    dumps = dumps_msgpack
else:
    loads = loads_pyarrow
    dumps = dumps_pyarrow
