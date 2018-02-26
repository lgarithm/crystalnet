#!/bin/sh

DATA_DIR=$HOME/var/data/mnist

download_mnist_data(){
    [ ! -f "$1" ] && curl -sOJ http://yann.lecun.com/exdb/mnist/$1
}

mkdir -p $DATA_DIR && cd $_

download_mnist_data train-images-idx3-ubyte.gz
download_mnist_data train-labels-idx1-ubyte.gz
download_mnist_data t10k-images-idx3-ubyte.gz
download_mnist_data t10k-labels-idx1-ubyte.gz

gzip -dfk *.gz
