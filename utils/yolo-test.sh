#!/bin/sh

set -ex

cd $(dirname $0)/..
pwd

# DARKNET_SRC=$HOME/code/mirrors/github.com/pjreddie/darknet
DARKNET_SRC=$HOME/code/repos/gitee.com/kuroicrow/darknet
# TEST_IMAGE=$DARKNET_SRC/data/dog.jpg
# TEST_IMAGE=$DARKNET_SRC/sized.bmp
TEST_IMAGE=$DARKNET_SRC/sized.idx

MODEL_DIR=$HOME/var/models/yolo

LOG_DIR=$(pwd)/log
LOG_FILE=$LOG_DIR/crystalnet.log
YOLO_BIN=$(pwd)/build/$(uname -s)/bin/yolo

[ ! -d $LOG_DIR ] && git init $LOG_DIR

time $YOLO_BIN -m $MODEL_DIR -f $TEST_IMAGE 2>stderr.log | tee $LOG_FILE
EXIT_CODE=$?
echo "log saved to $LOG_FILE , exit code: $EXIT_CODE"
