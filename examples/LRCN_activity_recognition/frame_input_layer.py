#!/usr/bin/env python

#Data layer for video.  Change flow_frames and RGB_frames to be the path to the flow and RGB frames.

import sys
sys.path.append('../../python')
import caffe
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import time
import pdb
import glob
import pickle as pkl
import random
import h5py
from multiprocessing import Pool
from threading import Thread
import skimage.io
import copy
import scipy

flow_frames = 'flow_images/'
RGB_frames = 'frames'
test_buffer = 1
train_buffer = 64
crop_size = 224
image_height = 320
image_width = 240

def processImageCrop(im_info, transformer, flow):
  im_path = im_info[0]
  im_crop = im_info[1] 
  im_reshape = im_info[2]
  im_flip = im_info[3]
  im_rotate = im_info[4]
  data_in = caffe.io.load_image(im_path)

  if (data_in.shape[0] < im_reshape[0]) | (data_in.shape[1] < im_reshape[1]):
    data_in = caffe.io.resize_image(data_in, im_reshape)

  if im_flip:
    data_in = caffe.io.flip_image(data_in, 1, flow) 
    data_in = data_in[im_crop[0]:im_crop[2], im_crop[1]:im_crop[3], :]

  if im_rotate != 0.:
    rows = im_reshape[0]
    cols = im_reshape[1]
    data_in = scipy.ndimage.interpolation.rotate(data_in,im_rotate,reshape=False)

  processed_image = transformer.preprocess('data_in',data_in)
  return processed_image

class ImageProcessorCrop(object):
  def __init__(self, transformer, flow):
    self.transformer = transformer
    self.flow = flow
  def __call__(self, im_info):
    return processImageCrop(im_info, self.transformer, self.flow)

class sequenceGeneratorFrame(object):
  def __init__(self, buffer_size, num_frames, frame_dict, frame_order, augment=True):
    self.buffer_size = buffer_size
    self.num_frames = num_frames
    self.frame_dict = frame_dict
    self.frame_order = frame_order
    self.augment = augment
    self.idx = 0

  def __call__(self):
    label_r = []
    im_paths = []
    im_crop = []
    im_reshape = []  
    im_flip = []
    im_rotate = []
 
    if self.idx + self.buffer_size >= self.num_frames:
      idx_list = range(self.idx, self.num_frames)
      idx_list.extend(range(0, self.buffer_size-(self.num_frames-self.idx)))
    else:
      idx_list = range(self.idx, self.idx+self.buffer_size)
    
    for i in idx_list:
      key = self.frame_order[i]
      frame_path = self.frame_dict[key]['frame']
      label = self.frame_dict[key]['label']
      frame_reshape = self.frame_dict[key]['reshape']
      frame_crop = self.frame_dict[key]['crop']
      label_r.append(label)

      im_reshape.append((frame_reshape))
      r0 = int(random.random()*(frame_reshape[0] - frame_crop[0]))
      r1 = int(random.random()*(frame_reshape[1] - frame_crop[1]))
      im_crop.append((r0, r1, r0+frame_crop[0], r1+frame_crop[1]))

      f = 0
      theta = 0.
      if self.augment:
        f = random.randint(0,1)
        theta = np.random.normal(0., 30.)
      im_flip.append(f)
      im_rotate.append(theta)
      
      im_paths.append(frame_path) 
    
    im_info = zip(im_paths,im_crop,im_reshape,im_flip,im_rotate)

    self.idx += self.buffer_size
    if self.idx >= self.num_frames:
      self.idx = self.idx - self.num_frames

    return label_r, im_info
  
def advance_batch(result, sequence_generator, image_processor, pool):
    label_r, im_info = sequence_generator()
    tmp = image_processor(im_info[0])
    result['data'] = pool.map(image_processor, im_info)
    result['label'] = label_r
    # cm = np.ones(len(label_r))
    # cm[0::sequence_generator.clip_length] = 0
    # result['clip_markers'] = cm

class BatchAdvancer():
    def __init__(self, result, sequence_generator, image_processor, pool):
      self.result = result
      self.sequence_generator = sequence_generator
      self.image_processor = image_processor
      self.pool = pool
 
    def __call__(self):
      return advance_batch(self.result, self.sequence_generator, self.image_processor, self.pool)

class frameRead(caffe.Layer):

  def initialize(self):
    self.train_or_test = 'test'
    self.flow = False
    self.buffer_size = test_buffer  #num videos processed per batch
    self.idx = 0
    self.channels = 3
    self.height = crop_size
    self.width = crop_size
    self.path_to_images = RGB_frames 
    self.frame_list = 'ucf101_split1_testFrames.txt' 

  def setup(self, bottom, top):
    random.seed(10)
    self.initialize()
    f = open(self.frame_list, 'r')
    f_lines = f.readlines()
    f.close()

    frame_dict = {}
    current_line = 0
    self.frame_order = []
    for ix, line in enumerate(f_lines):
      frame = line.split(' ')[0]
      l = int(line.split(' ')[1])
      frame_path = '%s%s' % (self.path_to_images, frame)

      frame_dict[frame] = {}
      frame_dict[frame]['frame'] = frame_path
      frame_dict[frame]['reshape'] = (image_height,image_width)
      frame_dict[frame]['crop'] = (crop_size, crop_size)
      frame_dict[frame]['label'] = l
      self.frame_order.append(frame) 

    self.frame_dict = frame_dict
    self.num_frames = len(self.frame_dict.keys())

    #set up data transformer
    shape = (self.buffer_size, self.channels, self.height, self.width)
        
    self.transformer = caffe.io.Transformer({'data_in': shape})
    self.transformer.set_raw_scale('data_in', 255)
    if self.flow:
      image_mean = [128, 128, 128]
      self.transformer.set_is_flow('data_in', True)
    else:
      image_mean = [104, 117, 123]
      self.transformer.set_is_flow('data_in', False)
    channel_mean = np.zeros((3,crop_size,crop_size))
    for channel_index, mean_val in enumerate(image_mean):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data_in', channel_mean)
    self.transformer.set_channel_swap('data_in', (2, 1, 0))
    self.transformer.set_transpose('data_in', (2, 0, 1))

    self.thread_result = {}
    self.thread = None
    pool_size = 24
    augment = self.train_or_test == 'train'

    self.image_processor = ImageProcessorCrop(self.transformer, self.flow)
    self.sequence_generator = sequenceGeneratorFrame(self.buffer_size, self.num_frames, self.frame_dict, self.frame_order, augment=augment)

    self.pool = Pool(processes=pool_size)
    self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.image_processor, self.pool)
    self.dispatch_worker()

    self.top_names = ['data', 'label']
    print 'Outputs:', self.top_names

    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))

    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'data':
        shape = (self.buffer_size, self.channels, self.height, self.width)
      elif name == 'label':
        shape = (self.buffer_size,)
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    #rearrange the data: The LSTM takes inputs as [video0_frame0, video1_frame0,...] but the data is currently arranged as [video0_frame0, video0_frame1, ...]
    for top_index, name in zip(range(len(top)), self.top_names):
      if name == 'data':
        for i in range(self.buffer_size):
          top[top_index].data[i, ...] = self.thread_result['data'][i] 
      elif name == 'label':
        top[top_index].data[...] = self.thread_result['label']

    self.dispatch_worker()
      
  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def backward(self, top, propagate_down, bottom):
    pass

class frameReadTrain_flow(frameRead):

  def initialize(self):
    self.train_or_test = 'train'
    self.flow = True
    self.buffer_size = train_buffer  #num videos processed per batch
    self.idx = 0
    self.channels = 3
    self.height = crop_size
    self.width = crop_size
    self.path_to_images = flow_frames 
    self.frame_list = 'data/ucf101_split1_trainFrames.txt' 

class frameReadTest_flow(frameRead):

  def initialize(self):
    self.train_or_test = 'test'
    self.flow = True
    self.buffer_size = test_buffer  #num videos processed per batch
    self.idx = 0
    self.channels = 3
    self.height = crop_size
    self.width = crop_size
    self.path_to_images = flow_frames 
    self.frame_list = 'data/ucf101_split1_testFrames.txt' 

class frameReadTrain_RGB(frameRead):

  def initialize(self):
    self.train_or_test = 'train'
    self.flow = False
    self.buffer_size = train_buffer  #num videos processed per batch
    self.idx = 0
    self.channels = 3
    self.height = crop_size
    self.width = crop_size
    self.path_to_images = RGB_frames 
    self.frame_list = 'data/bu4dfe_frame_lab_test.txt' 

class frameReadTest_RGB(frameRead):

  def initialize(self):
    self.train_or_test = 'test'
    self.flow = False
    self.buffer_size = test_buffer  #num videos processed per batch
    self.idx = 0
    self.channels = 3
    self.height = crop_size
    self.width = crop_size
    self.path_to_images = RGB_frames 
    self.frame_list = 'data/bu4dfe_frame_lab_test.txt' 
