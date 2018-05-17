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
# File: utils.py

import six


class VariableHolder(object):
    """ A proxy to access variables defined in a layer. """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: {name:variable}
        """
        self._vars = {}
        for k, v in six.iteritems(kwargs):
            self._add_variable(k, v)

    def _add_variable(self, name, var):
        assert name not in self._vars
        self._vars[name] = var

    def __setattr__(self, name, var):
        if not name.startswith('_'):
            self._add_variable(name, var)
        else:
            # private attributes
            super(VariableHolder, self).__setattr__(name, var)

    def __getattr__(self, name):
        return self._vars[name]

    def all(self):
        """
        Returns:
            list of all variables
        """
        return list(six.itervalues(self._vars))
