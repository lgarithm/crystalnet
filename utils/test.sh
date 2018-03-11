#!/bin/sh

set -e
find tests/build/bin -type f -print -exec {} \;
