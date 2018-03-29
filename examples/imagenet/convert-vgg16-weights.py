#!/usr/bin/env python3
#
# convert vgg16_weights.npz
#
import os
from os import path
import numpy as np

import idx

model_dir = path.join(os.getenv('HOME'), 'var/models/vgg16')
ws = np.load(path.join(model_dir, 'vgg16_weights.npz'))

for i, name in enumerate(ws.files):
    w = ws[name]
    print('%-8d %-24s %-24s %s' % (i + 1, name, w.dtype, w.shape))
    idx.write_idx(path.join(model_dir, name) + '.idx', w)
