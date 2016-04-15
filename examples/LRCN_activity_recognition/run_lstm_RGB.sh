#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

GLOG_logtostderr=1  $TOOLS/caffe train -solver solver/lstm_solver_RGB.prototxt -weights singleFrame_RGB_iter_5000.caffemodel 
echo "Done."
