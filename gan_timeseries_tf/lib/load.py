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
from six.moves import xrange

from data_utils import BaseDataSet
#from data_utils import z_normalize

def load_data(filename, delimiter=','):
    """

    :param filename: 
    :param delimiter: 
    :return: 
        X: [number_of_sample, length]
        y: labels, [0,....]
    """
    data = np.loadtxt(filename, dtype=np.float32, delimiter=delimiter)
    X = data[:, 1::]
    y = data[:, 0]
    y = y.astype(np.int) - 1
    return X, y


def read_data(data_root, fname):
    file_train = data_root + '/' + fname + '/' + fname + '_TRAIN'
    file_test = data_root + '/' + fname + '/' + fname + '_TEST'

    X_train, y_train = load_data(file_train)
    X_test, y_test = load_data(file_test)

    n_class = len(np.unique(np.concatenate([y_train, y_test])))

    dataset = BaseDataSet(X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test,
                          n_class=n_class)
    return dataset


def sine_wave(seq_length=100, num_samples=28 * 5 * 100, num_signals=1,
              freq_low=1, freq_high=5, amplitude_low=0.1, amplitude_high=0.9, **kwargs):
    """copy from data_utils/sine_wave"""
    ix = np.arange(seq_length) + 1
    samples = []
    for i in xrange(num_samples):
        signals = []
        for ii in xrange(num_signals):
            freq = np.random.uniform(low=freq_high, high=freq_low)  # frequency
            A = np.random.uniform(low=amplitude_low, high=amplitude_high)  # amplitude
            offset = np.random.uniform(low=-np.pi, high=np.pi)
            signals.append(A * np.sin(2 * np.pi * freq * ix / float(seq_length) + offset))
        samples.append(np.array(signals).T)
    # the shape of the samples is num_sample x seq_length x num_signals
    samples = np.array(samples)
    return samples
