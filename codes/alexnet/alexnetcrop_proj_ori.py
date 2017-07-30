import tensorflow as tf
import numpy as np
import random
import os

import math
from PIL import Image
from numpy import int32


import sys
sys.path.insert(0,'.')
sys.path.insert(0,'../imagenetdata')

from getimagenetclasses import *
from alexnet_proj_ori import *


def preproc_py2(imname,shorterside):
  
  
  pilimg = Image.open(imname)
  w,h=pilimg.size
  
  #print(w,h)

  if w > h:
    longerside= np.int32(math.floor(float(shorterside)/float(h)*w))
    neww=longerside
    newh=shorterside
  else:
    longerside= np.int32(math.floor(float(shorterside)/float(w)*h))
    newh=longerside
    neww=shorterside    
  resimg=pilimg.resize((neww,newh))
  
  
  im = np.array(resimg,dtype=np.float32)


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
    if current_h+hsize > h or current_w+wsize > w:
      break
    patches.append(im[int(current_h):int(current_h+hsize),int(current_w):int(current_w+wsize),:])
    current_h += sliding_window_h
    current_w += sliding_window_w
  return patches

def preproc(image):
  
  '''
  filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./images/*.jpg"))

  # Read an entire image file which is required since they're JPEGs, if the images
  # are too large they could be split in advance to smaller files or use the Fixed
  # reader to split up the file.
  image_reader = tf.WholeFileReader()
  
  # Read a whole file from the queue, the first returned value in the tuple is the
  # filename which we are ignoring.
  _, image_file = image_reader.read(filename_queue)
  
  '''
  
  height=tf.shape(image)[0]
  width=tf.shape(image)[1]
  new_shorter_edge=tf.cast(64, tf.int32)
  
  def _compute_longer_edge(height, width, new_shorter_edge):
    return tf.cast(width*new_shorter_edge/height, tf.int32)
  
  
  
  height_smaller_than_width = tf.less_equal(height, width)
  new_height_and_width = tf.cond(
    height_smaller_than_width,
    lambda: (new_shorter_edge, _compute_longer_edge(height, width, new_shorter_edge)),
    lambda: (_compute_longer_edge(width, height, new_shorter_edge), new_shorter_edge)
  )
  
  
  if image.dtype != tf.float32:
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  
    
  resimg=tf.image.resize_images(
      image,
      new_height_and_width
      #tf.stack(new_height_and_width) #,
      #tf.concat(new_height_and_width)
      #method=ResizeMethod.BILINEAR,
      #align_corners=False
  )
  resimg = tf.expand_dims(resimg, 0)
  #image = tf.subtract(image, 110.0)

  return resimg



class datagetter:
  def __init__(self,synsetfile,impath,xmlpath,ending):
    pass
    self.indicestosynsets,self.synsetstoindices,self.synsetstoclassdescr=parsesynsetwords(synsetfile)
    self.imagelist=[]
    self.xmlpath=xmlpath
    self.ending=ending
    self.counter=0
    
    for root, dirs, files in os.walk(impath):
      for f in files:
        fname= os.path.join(root, f)
        if fname.endswith(self.ending):
          self.imagelist.append(fname)   
          
    print('found',len(self.imagelist),'relevant files')
  
  def filenametoxml(self,fn):
    f=os.path.basename(fn)
    
    if not f.endswith(self.ending):
      print('not f.endswith(self.ending)')
      exit()
      
    f=f[:-len(self.ending)]+'.xml'
    f=os.path.join(self.xmlpath,f) 
    
    return f
    
  def get_next_batch(self,batchsize):
    
    imlist=  self.imagelist[self.counter:int(min(self.counter+batchsize,len(self.imagelist)))]
    self.counter+=batchsize

    # wrap around if at end of dataset
    diff=self.counter - len(self.imagelist)
    if(diff>0):
      imlist.extend(self.imagelist[0:diff] )
      self.counter=diff 
    elif diff==0: # exactly at the end, haha
      self.counter=0

    labels=-np.ones((len(imlist)))
    
    for ct,f in enumerate(imlist):
      xmlfile=self.filenametoxml(f)
      label,_=parseclasslabel(xmlfile,self.synsetstoindices)
      labels[ct]=int(label)
      
    return imlist,labels  

def run3(synsetfile,impath,xmlpath):
  
  
  num=500 # 500 or 200
  batchsize=2
  num_classes=1000

  keep_prob=1.
  skip_layer=[]
  is_training=False
  
  imagenet_mean = np.array([104., 117., 123.], dtype=np.float32) 
  
  cls=get_classes()
  dataclass=datagetter(synsetfile,impath,xmlpath,'.JPEG')
  
  sess = tf.Session()

  
  #imname='/home/binder/entwurf6/tfplaycpu/ai/alexnet/beach.jpg'
  #imname='/home/binder/entwurf6/tfplaycpu/ai/alexnet/poodle.png'
  #imname='/home/binder/entwurf6/tfplaycpu/ai/alexnet/quail227.JPEG'

  hsize = 64  ###############################################################
  wsize = 64  ###############################################################
  x = tf.placeholder(tf.float32, [batchsize, hsize, wsize, 3])
  net=AlexNet(x, keep_prob, num_classes, skip_layer, is_training, weights_path = 'DEFAULT')
  out=net.fc8
  
  
  init = tf.global_variables_initializer()
  sess.run(init)
  net.load_initial_weights(sess)
  
  

  top1corr=0
  top5corr=0
  for i in range(num):
  
    ims,lb= dataclass.get_next_batch(batchsize)

    #print (totalim.shape,lb.shape)
    #print (lb)
    for ct,imname in enumerate(ims):
      im=preproc_py2(imname,250)
      #print (im.shape)
      #print (imname)
      
      if(im.ndim<3):
        im=np.expand_dims(im,2)
        im=np.concatenate((im,im,im),2)
      
      if(im.shape[2]>3):
        im=im[:,:,0:3]

      num_patches = 10
      sliding_window_h = 16 ###############################################################
      sliding_window_w = 16 ###############################################################
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

    predict_valuess = []
    for crop in range(len(patches)):
      predict_valuess.append(sess.run(out, feed_dict={x: totalim[crop]}))
      # has shape batchsize,numclasses
      #print(predict_values.shape)
    predict_values = np.mean(predict_valuess,0)
  
    for ct in range(len(ims)):
    
      ind = np.argpartition(predict_values[ct,:], -5)[-5:] #get highest ranked indices to check top 5 error 
      index=np.argmax(predict_values[ct,:]) #get highest ranked index to check top 1 error
      print(ind,predict_values[ct,ind],index)
      

      if(index==lb[ct]):
        top1corr+=1.0/(num*batchsize) #times the number of crops
      if( lb[ct] in ind):
        top5corr+=1.0/(num*batchsize) #times the number of crops
  
  print('top-1 corr', top1corr) 
  print('top-5 corr', top5corr) 
  
  #print ('np.max(predict_values)', np.max(predict_values))
  #print ('classindex: ',np.argmax(predict_values))
  #print ('classlabel: ', cls[np.argmax(predict_values)])

if __name__=='__main__':
  #run2()
  #m=np.load('./ilsvrc_2012_mean.npy')
  #print(np.mean(np.mean(m,2),1))
  synsetfile='/Users/apple/Documents/study/term8/Artificial Intelligence/inClassCoding/data_ augmentation/alexnetcode/synset_words.txt'
  impath='/Users/apple/Documents/study/term8/Artificial Intelligence/inClassCoding/data_ augmentation/alexnetcode/imagespart/'
  xmlpath='/Users/apple/Documents/study/term8/Artificial Intelligence/inClassCoding/data_ augmentation/alexnetcode/val/'
  run3(synsetfile,impath,xmlpath)

