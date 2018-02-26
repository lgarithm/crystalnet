#!/bin/sh

set -e
find build/bin/*_test -print -exec {} \;

