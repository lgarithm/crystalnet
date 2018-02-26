#!/bin/sh

# https://www.cs.toronto.edu/~kriz/cifar.html

DATA=$HOME/var/data

download_cifar(){
    local prefix=https://www.cs.toronto.edu/~kriz/
    [ ! -f "$1" ] && curl -sOJ $prefix/$1
}

mkdir -p $DATA/cifar && cd $_

download_cifar cifar-10-binary.tar.gz
tar -xvf cifar-10-binary.tar.gz
