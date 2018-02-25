#!/bin/bash

set -e
find build/bin -type f -print -exec valgrind --leak-check=full -q --xtree-leak=yes {} \;
