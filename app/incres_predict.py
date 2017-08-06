# Credit to Kwot Sin's code on transfer learning and tensorflow/slim

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import time
import os
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# plt.style.use('ggplot')
slim = tf.contrib.slim

log_eval = './log_eval_test'
dataset_dir = '.'

batch_size = 1
num_epochs = 1
image_size = 299
num_classes = 2
num_samples = 1
split = 1

log_dir = './log' + str(split)
checkpoint_file = tf.train.latest_checkpoint(log_dir)

images_placeholder = tf.placeholder(tf.float32, [image_size, image_size, 3])
print ('Pictures loaded')

with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(images_placeholder, num_classes = num_classes, is_training = True)

variables_to_restore = slim.get_variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
def restore_fn(sess):
    return saver.restore(sess, checkpoint_file)

predictions = tf.argmax(end_points['Predictions'], 1)
print ('Tensors initialized')

sv = tf.train.Supervisor(logdir = None, summary_op = None, saver = None, init_fn = restore_fn)

def predict(input):

    with sv.managed_session() as sess:

        image = Image.open(input)
        image = image.resize((image_size,image_size))
        image = np.array(image)

        print ('Predicting...')
        prediction = sess.run(predictions, feed_dict={images_placeholder: image})
        print ('Predictions: ', prediction[0])

    

    return prediction[0]

    # text = 'Prediction: %s \n Ground Truth: %s' %(prediction[0], label)
    # img_plot = plt.imshow(image)

    # plt.title(text)
    # img_plot.axes.get_yaxis().set_ticks([])
    # img_plot.axes.get_xaxis().set_ticks([])
    # plt.show()


if __name__ == '__main__':
    predict(input = './BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549G/100X/SOB_B_A-14-22549G-100-019.png')




