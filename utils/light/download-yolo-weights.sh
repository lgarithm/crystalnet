#!/bin/sh

set -e

MODEL_DIR=$HOME/var/models/yolo

mkdir -p ${MODEL_DIR}
cd ${MODEL_DIR}

[ ! -f yolov2.weights ] && curl -vLOJ https://pjreddie.com/media/files/yolov2.weights
# TODO: validate checksum
