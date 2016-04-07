#!/bin/sh
TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

GLOG_logtostderr=1 $TOOLS/caffe train -solver solver/singleFrame_solver_RGB.prototxt -weights caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000 
echo 'Done.'
