#!/usr/bin/env python3
import os
from os import path

from scipy.misc import imread, imresize, imsave
import numpy as np

import idx

model_dir = path.join(os.getenv('HOME'), 'var/models/vgg16')
img = imread(path.join(model_dir, 'laska.png'))
print('%s%s' % (img.dtype, img.shape))

img = imresize(img, (224, 224))
print('%s%s' % (img.dtype, img.shape))

imsave(path.join(model_dir, 'laska.bmp'), img)
print('%s%s' % (img.dtype, img.shape))

img = imread(path.join(model_dir, 'laska.bmp'))  # remove alpha channel
print('%s%s' % (img.dtype, img.shape))

img = img.astype(np.float32)
print('%s%s' % (img.dtype, img.shape))

idx.write_idx(path.join(model_dir, 'laska.idx'), img)
