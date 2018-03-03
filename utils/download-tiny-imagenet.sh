#!/bin/sh

# https://tiny-imagenet.herokuapp.com/

DATA_DIR=$HOME/var/data/imagenet
mkdir -p $DATA_DIR && cd $DATA_DIR

download_tiny_imagenet(){
    local prefix=http://cs231n.stanford.edu
    [ ! -f "$1" ] && curl -sOJ $prefix/$1
}

download_tiny_imagenet tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
