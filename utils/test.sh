#!/bin/sh

set -e

for t in $(find tests/build/bin -type f); do
    echo "running $t"
    $t
done
