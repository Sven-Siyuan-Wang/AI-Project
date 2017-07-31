import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from numpy import int32
import random
import math
import time
from PIL import Image
from alexnet_proj import *

def random_patch(im,hsize,wsize,n):
  h = im.shape[0]
  w = im.shape[1]
  patches = []
  for i in range(n):
    ranh = random.randint(0,h-hsize)
    ranw = random.randint(0, w-hsize)
    patches.append(im[int(ranh):int(ranh+hsize),int(ranw):int(ranw+wsize),:])
  return patches

def sliding_patch(im,hsize,wsize,sliding_window_h,sliding_window_w):
  h = im.shape[0]
  w = im.shape[1]
  patches = []
  current_h = 0
  current_w = 0
  while True:
    if current_h+hsize <= h and current_w+wsize <= w:
      patches.append(im[int(current_h):int(current_h+hsize),int(current_w):int(current_w+wsize),:])

    if current_w+wsize > w and current_h+sliding_window_h+hsize <= h:
      current_h += sliding_window_h
      current_w = 0
    elif current_h+hsize <= h and current_w+sliding_window_w+wsize <= h:
      current_w += sliding_window_w
    else:
      break
  return patches

def load_image(path, split, res, part, BATCH_SIZE = 5):
  filename_train = path+'/split'+str(split)+'/'+str(res)+'X_train.txt'
  filename_val = path+'/split'+str(split)+'/'+str(res)+'X_val.txt'
  filename_test = path+'/split'+str(split)+'/'+str(res)+'X_test.txt'

  with open(filename_train) as f:
    train = f.readlines()
  train_labels = [int(x.strip().split(' ')[1]) for x in train]
  train_images = [path+x.strip().split(' ')[0] for x in train]
  print('Training set size: ' + str(len(train_images)))

  with open(filename_val) as f:
    val = f.readlines()
  val_labels = [int(x.strip().split(' ')[1]) for x in val]
  val_images = [path+x.strip().split(' ')[0] for x in val]
  print('Validation set size: ' + str(len(val_images)))

  with open(filename_val) as f:
    test = f.readlines()
  test_labels = [int(x.strip().split(' ')[1]) for x in test]
  test_images = [path+x.strip().split(' ')[0] for x in test]
  print('Test set size: ' + str(len(test_images)))

  NUM_CHANNELS = 3
  pilimg = Image.open(train_images[0])
  IMAGE_HEIGHT, IMAGE_WIDTH = pilimg.size

  if part == 'train':
    images = train_images
    labels = train_labels
  elif part == 'val':
    images = val_images
    labels = val_labels
  elif part == 'test':
    images = test_images
    labels = test_labels
  
  image_queue, label = tf.train.slice_input_producer([images, labels], shuffle=False)
  value = tf.read_file(image_queue)

  image = tf.image.decode_png(value, channels=NUM_CHANNELS)
  image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
  image.set_shape((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
  image_batch, label_batch = tf.train.batch([image, label],batch_size=BATCH_SIZE,capacity=1)

  return image_batch, label_batch, len(images)


def get_next_batch(sess, image_batch, label_batch):
    
    images = sess.run(image_batch)
    labels = sess.run(label_batch)

    return images, labels


def run_training(path='./BreaKHis_data/'):
  num_classes = 2
  epoch = 1

  num_patches = 260
  batchsize = 1
  keep_prob = 0.9
  skip_layer = []
  is_training = True

  imagenet_mean = np.array([185., 182., 188.], dtype=np.float32)

  train_image_batch, train_label_batch, train_set_size = load_image(path, 1, 200, 'train', BATCH_SIZE=batchsize)
  val_image_batch, val_label_batch, val_set_size = load_image(path, 1, 200, 'val', BATCH_SIZE=batchsize)
  hsize = 64
  wsize = 64
  sliding_window_h = 32
  sliding_window_w = 32

  x = tf.placeholder(tf.float32, [260, hsize, wsize, 3])
  net = AlexNet(x, keep_prob, num_classes, skip_layer, is_training, weights_path='DEFAULT')
  out = net.fc8
  labels_placeholder = tf.placeholder(tf.int64, shape=(1))
  pred = [tf.reduce_mean(net.fc8,axis=0)]
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,
                                                                 labels=labels_placeholder,
                                                                 name='cross-entropy')
  loss = tf.reduce_mean(cross_entropy, name='cross-entropy_mean', axis=0)
  optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
  train_op = optimizer.minimize(loss)


  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for j in range(epoch*train_set_size):
      tic = time.time()
      print ('\n======= No.', j,' out of ',epoch*train_set_size, '=======')
      next_image, next_label = get_next_batch(sess, train_image_batch, train_label_batch)
      print ('Next label: ', next_label)
      print ('Next image: ', next_image.shape)
      for ct, imname in enumerate(next_image):
        im = imname
        if (im.ndim < 3):
          im = np.expand_dims(im, 2)
          im = np.concatenate((im, im, im), 2)

        if (im.shape[2] > 3):
          im = im[:, :, 0:3]

        #patches = random_patch(im, hsize, wsize, num_patches)
        patches = sliding_patch(im, hsize, wsize, sliding_window_h, sliding_window_w)
        patches = patches - imagenet_mean
        print ('Patches: ', np.array(patches).shape)

      # predict_valuess = []
      # for crop in range(len(patches)):
        # predict_valuess.append(out)
      # predict_values = tf.reduce_mean(predict_valuess, axis=0)

      print ('Loss: ', sess.run([train_op, loss], feed_dict={labels_placeholder: next_label, x: patches}))
      print ('Time elapsed: ', time.time()-tic)

      # val_image, val_label = get_next_batch(sess, val_image_batch, val_label_batch)
      # for ct, imname in enumerate(next_image):
      #   im = imname
      #   if (im.ndim < 3):
      #     im = np.expand_dims(im, 2)
      #     im = np.concatenate((im, im, im), 2)

      #   if (im.shape[2] > 3):
      #     im = im[:, :, 0:3]

      #   #patches = random_patch(im, hsize, wsize, num_patches)
      #   patches = sliding_patch(im, hsize, wsize, sliding_window_h, sliding_window_w)
      #   patches = patches - imagenet_mean
      # print ('Validation label: ', val_label)
      # print ('Validation pred: \n', sess.run([pred], feed_dict={x: patches}))

    coord.request_stop()
    coord.join(threads)


def run_testing(path='./BreaKHis_data/'):
  num_classes = 2
  epoch = 1

  num_patches = 260
  batchsize = 1
  keep_prob = 0.9
  skip_layer = []
  is_training = True

  imagenet_mean = np.array([185., 182., 188.], dtype=np.float32)

  test_image_batch, test_label_batch, test_set_size = load_image(path, 1, 200, 'test', BATCH_SIZE=batchsize)
  #val_image_batch, val_label_batch, val_set_size = load_image(path, 1, 200, 'val', BATCH_SIZE=batchsize)
  hsize = 64
  wsize = 64
  sliding_window_h = 32
  sliding_window_w = 32

  x = tf.placeholder(tf.float32, [260, hsize, wsize, 3])
  net = AlexNet(x, keep_prob, num_classes, skip_layer, is_training, weights_path='DEFAULT')
  out = net.fc8
  #labels_placeholder = tf.placeholder(tf.int64, shape=(1))
  pred = [tf.reduce_mean(net.fc8,axis=0)]

  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for j in range(epoch*test_set_size):
      tic = time.time()
      print ('\n======= No.', j,' out of ',epoch*test_set_size, '=======')
      next_image, next_label = get_next_batch(sess, test_image_batch, test_label_batch)
      print ('Next label: ', next_label)
      print ('Next image: ', next_image.shape)
      for ct, imname in enumerate(next_image):
        im = imname
        if (im.ndim < 3):
          im = np.expand_dims(im, 2)
          im = np.concatenate((im, im, im), 2)

        if (im.shape[2] > 3):
          im = im[:, :, 0:3]

        #patches = random_patch(im, hsize, wsize, num_patches)
        patches = sliding_patch(im, hsize, wsize, sliding_window_h, sliding_window_w)
        patches = patches - imagenet_mean
        print ('Patches: ', np.array(patches).shape)

      # predict_valuess = []
      # for crop in range(len(patches)):
        # predict_valuess.append(out)
      # predict_values = tf.reduce_mean(predict_valuess, axis=0)

      print ('Loss: ', sess.run([out], feed_dict={x: patches}))
      print ('Time elapsed: ', time.time()-tic)

    coord.request_stop()
    coord.join(threads)

if __name__=='__main__':
  run_training(path='./BreaKHis_data/')
