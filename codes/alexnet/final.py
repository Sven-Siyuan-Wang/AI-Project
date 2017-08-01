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
  
  image_queue, label = tf.train.slice_input_producer([images, labels], shuffle=True)
  value = tf.read_file(image_queue)

  image = tf.image.decode_png(value, channels=NUM_CHANNELS)
  image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
  image.set_shape((IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
  image_batch, label_batch = tf.train.batch([image, label],batch_size=BATCH_SIZE,capacity=1)

  return image_batch, label_batch, len(images)


def get_next_batch(sess, image_batch, label_batch, batchsize, num_patches, hsize, wsize):

  sliding_window_h = 32
  sliding_window_w = 32 

  imagenet_mean = np.array([185., 182., 188.], dtype=np.float32)
  
  next_image = sess.run(image_batch)
  next_label = sess.run(label_batch)
  all_labels = np.repeat(next_label, num_patches)

  all_images = np.zeros((batchsize*num_patches,hsize,wsize,3))

  for ct, imname in enumerate(next_image):

      patches = random_patch(imname, hsize, wsize, num_patches)
      # patches = sliding_patch(imname, hsize, wsize, sliding_window_h, sliding_window_w)
      patches = patches - imagenet_mean

      all_images[ct*num_patches:(ct+1)*num_patches,:,:,:] = patches

  index = np.random.choice(range(batchsize*num_patches), batchsize*num_patches, replace=False)
  all_images = all_images[index]
  all_labels = all_labels[index]

  return all_images, all_labels


def run_training(path='./BreaKHis_data/', model_path='./model/model.ckpt'):

  num_classes = 2

  epoch = 10
  batchsize = 10
  hsize = 64
  wsize = 64
  num_patches = 500
  num_minibatch = 20

  minibatchsize = int(np.ceil(num_patches * batchsize / num_minibatch))
  keep_prob = 0.9
  skip_layer = []
  is_training = True


  train_image_batch, train_label_batch, train_set_size = load_image(path, 1, 200, 'train', BATCH_SIZE=batchsize)
  

  x = tf.placeholder(tf.float32, [minibatchsize, hsize, wsize, 3])
  labels_placeholder = tf.placeholder(tf.int64, shape=(minibatchsize))
  net = AlexNet(x, keep_prob, num_classes, skip_layer, is_training, weights_path='DEFAULT')
  y_pred = tf.nn.softmax(net.fc8)
  y_pred_cls = tf.argmax(y_pred, dimension=1)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net.fc8,
                                                                 labels=labels_placeholder,
                                                                 name='cross-entropy')
  loss = tf.reduce_mean(cross_entropy, name='cross-entropy_mean', axis=0)

  optimizer = tf.train.AdamOptimizer(learning_rate=0.000001, beta1=0.9, beta2=0.9, epsilon=0.0001)

  train_op = optimizer.minimize(loss)
  correct_prediction = tf.equal(y_pred_cls, labels_placeholder)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  saver = tf.train.Saver()



  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    iterPerEpoch = int(train_set_size/batchsize)
    iterations = epoch * iterPerEpoch

    tic = time.time()
    for j in range(iterations):
     
      print ('\n======= No.', j,' out of ',iterations, '=======')
      
      all_images, all_labels = get_next_batch(sess, train_image_batch, train_label_batch,
                                              batchsize, num_patches, hsize, wsize)

      for mini in range(num_minibatch):

        #tic = time.time()
        #print ('Minibatch ',mini)

        if (mini+1) * minibatchsize > len(all_images):
          mini_images = all_images[-minibatchsize:]
          mini_labels = all_labels[-minibatchsize:]
        else:
          mini_images = all_images[mini*minibatchsize : (mini+1)*minibatchsize]
          mini_labels = all_labels[mini*minibatchsize : (mini+1)*minibatchsize]


        sess.run(train_op, feed_dict={labels_placeholder: mini_labels, x: mini_images})
        acc = sess.run(accuracy, feed_dict={labels_placeholder: mini_labels, x: mini_images})
        msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
        print(msg.format(j + 1, acc))

      print('Time total: ', time.time() - tic)
      if(j%iterPerEpoch == 0):
        save_path = saver.save(sess, model_path+str(j/iterPerEpoch))

    coord.request_stop()
    coord.join(threads)


def run_testing(path='./BreaKHis_data/', model_path='./model/model.ckpt'):
  accuracy = 0
  num_classes = 2

  num_patches = 260
  batchsize = 1
  keep_prob = 0.9
  skip_layer = []
  is_training = True

  imagenet_mean = np.array([185., 182., 188.], dtype=np.float32)

  test_image_batch, test_label_batch, test_set_size = load_image(path, 1, 200, 'test', BATCH_SIZE=batchsize)
  hsize = 64
  wsize = 64
  sliding_window_h = 32
  sliding_window_w = 32

  x = tf.placeholder(tf.float32, [num_patches, hsize, wsize, 3])
  net = AlexNet(x, keep_prob, num_classes, skip_layer, is_training, weights_path='DEFAULT')
  #labels_placeholder = tf.placeholder(tf.int64, shape=(1))
  prob = tf.nn.softmax(net.fc8)
  pred = [tf.reduce_mean(prob,axis=0)]

  with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    # restore session weights
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print("Model restored from file: %s" % model_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for j in range(test_set_size):
      tic = time.time()

      print ('\n======= No.', j,' out of ',test_set_size, '=======')
      next_image = sess.run(test_image_batch)
      next_label = sess.run(test_label_batch)
      for ct, imname in enumerate(next_image):

        #patches = random_patch(imname, hsize, wsize, num_patches)
        patches = sliding_patch(imname, hsize, wsize, sliding_window_h, sliding_window_w)
        patches = patches - imagenet_mean
        print ('Patches: ', np.array(patches).shape)

        pred_label = np.argmax(sess.run([pred], feed_dict={x: patches})[0][0])
        print("predicted lable: ", pred_label)

        if(next_label[0]==pred_label):
          accuracy += 1

      print ('Time elapsed: ', time.time()-tic)

    coord.request_stop()
    coord.join(threads)
    print("Total test set accuracy for ", test_set_size, " samples is: ", accuracy/test_set_size)
    

if __name__=='__main__':
  run_training(path='./BreaKHis_data/')
  tf.reset_default_graph()
  run_testing(path='./BreaKHis_data/')
