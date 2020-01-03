from __future__ import division
from __future__ import print_function
import os, scipy.io, scipy.misc, time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import tifffile
import glob
from PIL import Image

import model
import attention_unet

input_dir = './dataset/'
gt_dir = './dataset/long/'

#load the checkpoin of training model
checkpoint_dir = './checkpoint/'

result_dir = './resultdir/'

# get test IDs
test_fns = glob.glob(gt_dir + '/1*.tiff')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 3])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

out_image = attention_unet.AUnet_network(in_image, 3)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded', checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

#load the image for testing 
in_files = glob.glob(input_dir + '*.tiff')
for k in range(len(in_files)):
    in_path = in_files[k]
    in_fn = os.path.basename(in_path)
    in_fn1 = os.path.splitext(in_fn)[0]

    ratio = 2000
    st = time.time()

    in_img = tifffile.imread(in_path)
    input_full = np.expand_dims(np.float32(in_img/65535.0),axis = 0) * ratio
    
    input_full = np.minimum(input_full, 1.0)

    output = sess.run(out_image, feed_dict={in_image: input_full})
    output = np.minimum(np.maximum(output, 0), 1)

    output = output[0, :, :, :]

    scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
        result_dir + in_fn1 +'_out.png')

    print(in_fn, time.time()-st)
