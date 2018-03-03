#!/bin/sh

set -e
find tests/build/bin -type f -print -exec valgrind --leak-check=full -q --xtree-leak=yes {} \;
