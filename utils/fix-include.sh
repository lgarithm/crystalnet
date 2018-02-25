#!/bin/bash

iwyu -std=c++1z -I./src src/misaka/core/tensor.cpp 2>&1 | fix_include -b --comments
