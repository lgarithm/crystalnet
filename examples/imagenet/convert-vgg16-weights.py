#!/usr/bin/env python3
#
# convert vgg16_weights.npz
#
import os
from os import path
from struct import pack

import numpy as np


def idx_type(np_type):
    # https://docs.scipy.org/doc/numpy/user/basics.types.html
    if np_type == np.uint8:
        return 0x8
    elif np_type == np.int8:
        return 0x9
    elif np_type == np.uint16:
        return 0xb
    elif np_type == np.uint32:
        return 0xc
    if np_type == np.float32:
        return 0xd
    elif np_type == np.float64:
        return 0xe
    raise ValueError('unsupported dtype %s' % np_type)


def write_idx_header(f, a):
    f.write(pack('BBBB', 0, 0, idx_type(a.dtype), len(a.shape)))
    for dim in a.shape:
        f.write(pack('>I', dim))


def write_idx(name, a):
    print('saving to %s' % name)
    with open(name, 'wb') as f:
        write_idx_header(f, a)
        f.write(a.tobytes())

model_dir = path.join(os.getenv('HOME'), 'var/models/vgg16')
ws = np.load(path.join(model_dir, 'vgg16_weights.npz'))

for idx, name in enumerate(ws.files):
    w = ws[name]
    print('%-8d %-24s %-24s %s' % (idx + 1, name, w.dtype, w.shape))
    write_idx(path.join(model_dir, name) + '.idx', w)
