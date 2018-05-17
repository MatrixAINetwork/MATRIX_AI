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
# -*- coding: UTF-8 -*-
# File: _test.py


import sys
import cv2
from . import AugmentorList
from .crop import *
from .imgproc import *
from .noname import *
from .deform import *
from .noise import SaltPepperNoise


anchors = [(0.2, 0.2), (0.7, 0.2), (0.8, 0.8), (0.5, 0.5), (0.2, 0.5)]
augmentors = AugmentorList([
    Contrast((0.8, 1.2)),
    Flip(horiz=True),
    GaussianDeform(anchors, (360, 480), 0.2, randrange=20),
    # RandomCropRandomShape(0.3),
    SaltPepperNoise()
])

img = cv2.imread(sys.argv[1])
newimg, prms = augmentors._augment_return_params(img)
cv2.imshow(" ", newimg.astype('uint8'))
cv2.waitKey()

newimg = augmentors._augment(img, prms)
cv2.imshow(" ", newimg.astype('uint8'))
cv2.waitKey()
