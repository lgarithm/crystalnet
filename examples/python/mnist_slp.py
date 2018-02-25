#!/usr/bin/env python3

# TODO: implenent

import os
import sys
path = os.path.join(os.path.dirname(__file__), '../..', 'langs/python')
sys.path.append(path)
import misaka

for x in dir(misaka):
    print(x)
print(misaka.version())
shape = misaka.Shape(3)
print(shape)
