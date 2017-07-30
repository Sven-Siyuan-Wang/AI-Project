import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import random
import os

import math
from PIL import Image
from numpy import int32

from alexnet_proj import *


def preproc_py2(imname,shorterside):
  pilimg = Image.open(imname)

  return pilimg

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

  return image_batch, label_batch


def get_next_batch(sess, image_batch, label_batch):
    
    images = sess.run(image_batch)
    labels = sess.run(label_batch)

    return images, labels


def run3(path='./BreaKHis_data'):
  
  num=500 # 500 or 200
  batchsize=5
  num_classes=2

  keep_prob=1.
  skip_layer=[]
  is_training=False
  
  imagenet_mean = np.array([104., 117., 123.], dtype=np.float32) 
  
  train_image_batch, train_label_batch = load_image(path, 1, 200, 'train', BATCH_SIZE=batchsize)

  with tf.Session() as sess:
    hsize = 64  ###############################################################
    wsize = 64  ###############################################################
    x = tf.placeholder(tf.float32, [batchsize, hsize, wsize, 3])
    net = AlexNet(x, keep_prob, num_classes, skip_layer, is_training, weights_path = 'DEFAULT')
    out = net.fc8
    
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #net.load_initial_weights(sess)
    
    

    top1corr=0
    top5corr=0
    for i in range(num):
    
      next_image,next_label = get_next_batch(sess, train_image_batch, train_label_batch)
      #print (totalim.shape,lb.shape)
      #print (lb)
      for ct,imname in enumerate(next_image):
        #print(ct,imname.shape)
        #im=preproc_py2(imname,250)
        im = imname
        #print (im.shape)
        #print (imname)
        
        if(im.ndim<3):
          im=np.expand_dims(im,2)
          im=np.concatenate((im,im,im),2)
        
        if(im.shape[2]>3):
          im=im[:,:,0:3]

        num_patches = 10
        sliding_window_h = 32 ###############################################################
        sliding_window_w = 32 ###############################################################
        #patches = random_patch(im, hsize, wsize, num_patches)
        patches = sliding_patch(im, hsize, wsize, sliding_window_h, sliding_window_w)
        totalim = []
        for numbercrops in range(len(patches)):
          totalim.append(np.zeros((batchsize, hsize, wsize, 3)))
        #here need to average over 5 crops instead of one
        for crop in range(len(patches)):
          imcropped = patches[crop]
          imcropped=imcropped[:,:,[2,1,0]] #RGB to BGR
          imcropped=imcropped-imagenet_mean

          #print (imcropped.shape,totalim[crop].shape)
          totalim[crop][ct,:,:,:]=imcropped
        #print(np.array(totalim).shape)
      predict_valuess = []
      for crop in range(len(patches)):
        predict_valuess.append(sess.run(out, feed_dict={x: totalim[crop]}))
        # has shape batchsize,numclasses
      #print (len(patches))
      #print(np.array(predict_valuess).shape)
      predict_values = np.mean(predict_valuess,axis=0)

      '''
      for ct in range(len(next_image)):
      
        #ind = np.argpartition(predict_values[ct,:], -5)[-5:] #get highest ranked indices to check top 5 error
        index=np.argmax(predict_values[ct,:]) #get highest ranked index to check top 1 error
        #print(ind,predict_values[ct,ind],index)
        

        if(index==next_label[ct]):
          top1corr+=1.0/(num*batchsize) #times the number of crops

        #if( next_label[ct] in ind):
          #top5corr+=1.0/(num*batchsize) #times the number of crops
      '''
    coord.request_stop()
    coord.join(threads)
  return None

  #print ('np.max(predict_values)', np.max(predict_values))
  #print ('classindex: ',np.argmax(predict_values))
  #print ('classlabel: ', cls[np.argmax(predict_values)])


def run_training(path='./BreaKHis_data'):
  num = 500  # 500 or 200
  batchsize = 260 ###
  num_classes = 2

  num_patches = 5
  batchsize = 260
  keep_prob = 1.
  skip_layer = []
  is_training = True

  imagenet_mean = np.array([104., 117., 123.], dtype=np.float32)

  train_image_batch, train_label_batch = load_image(path, 1, 200, 'train', BATCH_SIZE=batchsize)
  hsize = 64  ###############################################################
  wsize = 64  ###############################################################
  sliding_window_h = 32  ###############################################################
  sliding_window_w = 32  ###############################################################

  x = tf.placeholder(tf.float32, [batchsize, hsize, wsize, 3])
  net = AlexNet(x, keep_prob, num_classes, skip_layer, is_training, weights_path='DEFAULT')
  out = net.fc8
  labels_placeholder = tf.placeholder(tf.int64, shape=(1))
  #preds_placeholder = tf.placeholder(tf.float64, shape=(batchsize))
  pred = [tf.reduce_mean(net.fc8,axis=0)]
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,
                                                                 labels=labels_placeholder,
                                                                 name='cross-entropy')
  loss = tf.reduce_mean(cross_entropy, name='cross-entropy_mean', axis=0)

  optimizer = tf.train.AdamOptimizer(learning_rate=0.000001, beta1=0.9, beta2=0.9, epsilon=(1.0 / 2014.0),
                                     name='Adam')
  train_op = optimizer.minimize(loss)

  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(num):

      next_image, next_label = get_next_batch(sess, train_image_batch, train_label_batch)
      for ct, imname in enumerate(next_image):
        im = imname
        if (im.ndim < 3):
          im = np.expand_dims(im, 2)
          im = np.concatenate((im, im, im), 2)

        if (im.shape[2] > 3):
          im = im[:, :, 0:3]

        #patches = random_patch(im, hsize, wsize, num_patches)
        patches = sliding_patch(im, hsize, wsize, sliding_window_h, sliding_window_w)
        totalim = []
        for numbercrops in range(len(patches)):
          totalim.append(np.zeros((batchsize, hsize, wsize, 3)))
        # here need to average over 5 crops instead of one
        for crop in range(len(patches)):
          imcropped = patches[crop]
          imcropped = imcropped[:, :, [2, 1, 0]]  # RGB to BGR
          imcropped = imcropped - imagenet_mean
          totalim[crop][ct, :, :, :] = imcropped
      predict_valuess = []
      for crop in range(len(patches)):
        predict_valuess.append(out)
      predict_values = tf.reduce_mean(predict_valuess, axis=0)

      totalim2 = np.zeros((len(patches), hsize, wsize, 3))
      for i in range(len(totalim)):
        totalim2[i,:,:,:]= totalim[i][0,:,:,:]

      print(sess.run([train_op, loss], feed_dict={labels_placeholder: [next_label[0]], x: totalim2}))

    coord.request_stop()
    coord.join(threads)


if __name__=='__main__':
  #run3(path='/Users/apple/BreaKHis_data/')
  run_training(path='/Users/apple/BreaKHis_data/')
