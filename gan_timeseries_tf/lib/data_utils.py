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
import collections
import os

from sklearn import utils as skutils
from rng import np_rng, py_rng


BaseDataSet = collections.namedtuple('BaseDataSet',
                                     ['X_train', 'y_train',
                                      'X_test', 'y_test',
                                      'n_class'])


class DataSet(object):
    """refer tensorflow.examples.tutorials.mnist.input_data"""
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.num_examples = X.shape[0]
        self.epochs_completed = 0
        self.index_in_epoch = 0

    def next_batch(self, batch_size, is_shuffle=True):
        if self.y is None:
            return self._next_batch_X(batch_size, is_shuffle)
        else:
            return self._next_batch_X_y(batch_size, is_shuffle)

    def _next_batch_X(self, batch_size, is_shuffle=True):
        start = self.index_in_epoch
        # shuffle for the first epoch
        if self.epochs_completed == 0 and start == 0 and is_shuffle:
            self.X = shuffle(self.X)
        # Go to the next batch
        if start + batch_size > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Get the rest samples in this epoch
            rest_num_examples = self.num_examples - start
            X_rest_part = self.X[start:self.num_examples]
            # Shuffle the data
            if is_shuffle:
                self.X = shuffle(self.X)
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.X[start:end]
            return np.concatenate([X_rest_part, X_new_part], axis=0)
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            batch = self.X[start:end]
            return batch

    def _next_batch_X_y(self, batch_size, is_shuffle=True):
        start = self.index_in_epoch
        # shuffle for the first epoch
        if self.epochs_completed == 0 and start == 0 and is_shuffle:
            self.X = shuffle(self.X)
            self.y = shuffle(self.y)
        # Go the next batch
        if start + batch_size > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Get the rest samples in this epoch
            rest_num_example = self.num_examples - start
            X_rest_part = self.X[start:self.num_examples]
            y_rest_part = self.y[start:self.num_examples]
            # Shuffle the data
            if is_shuffle:
                self.X = shuffle(self.X)
                self.y = shuffle(self.y)
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_example
            end = self.index_in_epoch
            X_new_part = self.X[start:end]
            y_new_part = self.y[start:end]

            X_batch = np.concatenate([X_rest_part, X_new_part], axis=0)
            y_batch = np.concatenate([y_rest_part, y_new_part], axis=0)
            return X_batch, y_batch
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            X_batch = self.X[start:end]
            y_batch = self.y[start:end]
            return X_batch, y_batch


def shuffle_list(*data):
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]

def shuffle(*arrays, **options):
    if isinstance(arrays[0][0], str):
        return shuffle_list(*arrays)
    else:
        return skutils.shuffle(*arrays, random_state=np_rng)


def dense_to_one_hot(labels_dense, num_classes):
  """
    Convert class labels from scalars to one-hot vectors.
    copy from tensorflow.examples.tutorials.mnist.input_data
  """
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def scale_image(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min()) / (255 - x.min()))

    # scale to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x


def z_normalize(data):
    norm_data = data.copy()
    mean = np.mean(norm_data)
    std = np.std(norm_data)

    norm_data = norm_data - mean
    # The 1e-9 avoids dividing by zero
    norm_data = norm_data / ( std + 1e-9)

    return norm_data