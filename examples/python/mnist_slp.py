#!/usr/bin/env python3

# TODO: implenent

import os
import sys
path = os.path.join(os.path.dirname(__file__), '../..', 'langs/python')
sys.path.append(path)
import crystalnet as c

print(c.version())
image_shape = c.Shape(28, 28)
print(image_shape)


def slp(image_shape: c.Shape, arity: int):
    x = c.var(image_shape)
    x_ = c.reshape(x, c.Shape(image_shape.dim()))
    w = c.covar(c.Shape(image_shape.dim(), arity))
    b = c.covar(c.Shape(arity))
    y = c.apply(c.mul, x_, w)
    z = c.apply(c.add, y, b)
    o = c.apply(c.softmax, z)
    return c.Model(x, o)


# m = slp(image_shape, 10)
