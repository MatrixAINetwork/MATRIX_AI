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
# File: paste.py


from .base import ImageAugmentor

from abc import abstractmethod
import numpy as np

__all__ = ['CenterPaste', 'BackgroundFiller', 'ConstantBackgroundFiller',
           'RandomPaste']


class BackgroundFiller(object):
    """ Base class for all BackgroundFiller"""

    def fill(self, background_shape, img):
        """
        Return a proper background image of background_shape, given img.

        Args:
            background_shape (tuple): a shape (h, w)
            img: an image
        Returns:
            a background image
        """
        background_shape = tuple(background_shape)
        return self._fill(background_shape, img)

    @abstractmethod
    def _fill(self, background_shape, img):
        pass


class ConstantBackgroundFiller(BackgroundFiller):
    """ Fill the background by a constant """

    def __init__(self, value):
        """
        Args:
            value (float): the value to fill the background.
        """
        self.value = value

    def _fill(self, background_shape, img):
        assert img.ndim in [3, 2]
        if img.ndim == 3:
            return_shape = background_shape + (img.shape[2],)
        else:
            return_shape = background_shape
        return np.zeros(return_shape, dtype=img.dtype) + self.value


class CenterPaste(ImageAugmentor):
    """
    Paste the image onto the center of a background canvas.
    """

    def __init__(self, background_shape, background_filler=None):
        """
        Args:
            background_shape (tuple): shape of the background canvas.
            background_filler (BackgroundFiller): How to fill the background. Defaults to zero-filler.
        """
        if background_filler is None:
            background_filler = ConstantBackgroundFiller(0)

        self._init(locals())

    def _augment(self, img, _):
        img_shape = img.shape[:2]
        assert self.background_shape[0] >= img_shape[0] and self.background_shape[1] >= img_shape[1]

        background = self.background_filler.fill(
            self.background_shape, img)
        y0 = int((self.background_shape[0] - img_shape[0]) * 0.5)
        x0 = int((self.background_shape[1] - img_shape[1]) * 0.5)
        background[y0:y0 + img_shape[0], x0:x0 + img_shape[1]] = img
        return background

    def _augment_coords(self, coords, param):
        raise NotImplementedError()


class RandomPaste(CenterPaste):
    """
    Randomly paste the image onto a background convas.
    """

    def _get_augment_params(self, img):
        img_shape = img.shape[:2]
        assert self.background_shape[0] > img_shape[0] and self.background_shape[1] > img_shape[1]

        y0 = self._rand_range(self.background_shape[0] - img_shape[0])
        x0 = self._rand_range(self.background_shape[1] - img_shape[1])
        return int(x0), int(y0)

    def _augment(self, img, loc):
        x0, y0 = loc
        img_shape = img.shape[:2]
        background = self.background_filler.fill(
            self.background_shape, img)
        background[y0:y0 + img_shape[0], x0:x0 + img_shape[1]] = img
        return background
