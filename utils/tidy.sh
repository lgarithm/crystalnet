#!/bin/bash

cd build && find ../src -type f -name '*.cpp' -exec clang-tidy {} \;
