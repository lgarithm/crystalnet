#!/bin/sh

set -e
find tests/build/bin/*_test -print -exec {} \;
