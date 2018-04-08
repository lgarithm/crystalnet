#!/bin/sh

set -e

mkdir -p tests/build/results

for t in $(find tests/build/$(uname)/bin -type f); do
    echo "checking $t"
    result=tests/build/results/$(basename $t).result.xml
    valgrind --xml=yes --xml-file=${result} -q --leak-check=full ${t}
    ./utils/analysis-valgrind-result.rb ${result}
done
