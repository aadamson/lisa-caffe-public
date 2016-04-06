#!/bin/sh
TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe train -solver solver/singleFrame_solver_goog.prototxt
echo 'Done.'
