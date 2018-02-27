#!/bin/sh

# https://www.cs.toronto.edu/~kriz/cifar.html

DATA_DIR=$HOME/var/data/cifar

download_cifar(){
    local prefix=https://www.cs.toronto.edu/~kriz
    [ ! -f "$1" ] && curl -sOJ $prefix/$1
}

mkdir -p $DATA_DIR && cd $DATA_DIR

download_cifar cifar-10-binary.tar.gz
tar -xvf cifar-10-binary.tar.gz
