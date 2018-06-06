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
@author: Steve Deng
"""
import numpy as np

def table_row(row_elem):
    row_str = ""
    for elem in row_elem:
        row_str += "{} |".format(elem)
    row_str += "\n"
    return row_str

def table_head_line(n):
    str_split = ""
    for _ in np.arange(n):
        str_split += '---|'
    str_split += "\n"
    return str_split