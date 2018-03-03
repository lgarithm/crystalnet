#!/usr/bin/env python3

# TODO: implenent

import os
import sys
path = os.path.join(os.path.dirname(__file__), '../..', 'langs/python')
sys.path.append(path)
import crystalnet

for x in dir(crystalnet):
    print(x)
print(crystalnet.version())
shape = crystalnet.Shape(3)
print(shape)
