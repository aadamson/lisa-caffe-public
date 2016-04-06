#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

GLOG_logtostderr=1  $TOOLS/caffe train -solver solver/lstm_solver_RGB.prototxt -weights snapshots_singleFrame_RGB_iter_500.caffemodel 
echo "Done."
