import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import math
from PIL import Image
from numpy import int32
import copy
import sys

from alexnet import * 


def load_image(split, res):
  filename_train = './BreaKHis_data/split'+str(split)+'/'+str(res)+'X_train.txt'
  filename_val = './BreaKHis_data/split'+str(split)+'/'+str(res)+'X_val.txt'
  filename_test = './BreaKHis_data/split'+str(split)+'/'+str(res)+'X_test.txt'

  with open(filename_train) as f:
    train = f.readlines()
  train_labels = [int(x.strip().split(' ')[1]) for x in train]
  train_images = ['./BreaKHis_data/'+x.strip().split(' ')[0] for x in train]
  print('Training set size: ' + str(len(train_images)))

  with open(filename_train) as f:
    val = f.readlines()
  val_y = [int(x.strip().split(' ')[1]) for x in val]
  val_x = [x.strip().split(' ')[0] for x in val]

  with open(filename_train) as f:
    test = f.readlines()
  test_y = [int(x.strip().split(' ')[1]) for x in test]
  test_x = [x.strip().split(' ')[0] for x in test]

  NUM_CHANNELS = 3
  BATCH_SIZE = 5
  pilimg = Image.open(train_images[0])
  IMAGE_HEIGHT, IMAGE_WIDTH = pilimg.size

  train_image_queue, train_label = tf.train.slice_input_producer([train_images, train_labels], shuffle=False)
  value = tf.read_file(train_image_queue)

  train_image = tf.image.decode_png(value, channels=NUM_CHANNELS)
  train_image = tf.image.resize_images(train_image, [IMAGE_HEIGHT, IMAGE_WIDTH])
  train_image.set_shape((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
  train_image_batch, train_label_batch = tf.train.batch([train_image, train_label],batch_size=BATCH_SIZE,capacity=1)

  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(int(len(train_images)/BATCH_SIZE)):
      sess.run(train_image_batch)
      sess.run(train_label_batch)

    coord.request_stop()
    coord.join(threads)

load_image(1, 200)


