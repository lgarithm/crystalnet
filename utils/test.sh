#!/bin/sh

set -e

for t in $(find tests/build/$(uname)/bin -type f); do
    echo "running $t"
    $t
done
