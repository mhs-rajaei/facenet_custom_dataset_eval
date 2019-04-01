"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
# import facenet
# import lfw
import os
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

from os.path import join


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

log_path = os.path.join(PROJECT_PATH, 'output')
models_path = os.path.join(PROJECT_PATH, 'models')
# train_dataset_path = r'F:\Documents\JetBrains\PyCharm\OFR\images\1024First_lfw_160'
# eval_pairs_path = os.path.join(PROJECT_PATH, 'data/All_VIS_112_pairs_3.txt')


from importlib.machinery import SourceFileLoader
facenet = SourceFileLoader('facenet', os.path.join(PROJECT_PATH, 'MR_facenet.py')).load_module()

eval_data_reader = SourceFileLoader('eval_data_reader', os.path.join(PROJECT_PATH, 'eval_data_reader.py')).load_module()
verification = SourceFileLoader('verification', os.path.join(PROJECT_PATH, 'verification.py')).load_module()
lfw = SourceFileLoader('lfw', os.path.join(PROJECT_PATH, 'lfw.py')).load_module()


class Args:
    lfw_nrof_folds = 10
    distance_metric = 1
    use_flipped_images = True
    subtract_mean = True
    use_fixed_image_standardization = True

    net_depth = 50
    epoch = 1000
    lr_steps = [40000, 60000, 80000]
    momentum = 0.9
    weight_decay = 5e-4

    # image_size = [160, 160]
    num_output = 85164

    # train_dataset_dir = train_dataset_path
    train_dataset_dir = None
    summary_path = join(log_path, 'summary')
    ckpt_path = join(log_path, 'ckpt')
    log_file_path = join(log_path, 'logs')

    saver_maxkeep = 10
    buffer_size = 10000
    log_device_mapping = False
    summary_interval = 100
    ckpt_interval = 100
    validate_interval = 100
    show_info_interval = 100
    seed = 313
    nrof_preprocess_threads = 4

    # eval_dataset = eval_dir_path
    eval_pair = os.path.join(PROJECT_PATH, 'data/First_100_ALL VIS_160_1.txt')
    eval_dataset = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\First_100_ALL VIS_160"
    lfw_dir = eval_dataset
    batch_size = 32
    lfw_batch_size = batch_size
    model = os.path.join(PROJECT_PATH, 'models/facenet/20180402-114759')
    image_size = 160
    lfw_pairs = eval_pair


def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            pairs = lfw.read_pairs_2(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths, actual_issame = lfw.get_paths_2(os.path.expanduser(args.lfw_dir), pairs)

            # InsightFace_TF evaluation
            ver_list = []
            ver_name_list = []
            print('Begin evaluation %s convert.' % args.eval_dataset)

            data_set = eval_data_reader.load_eval_datasets_2(args)
            ver_list.append(data_set)
            ver_name_list.append(args.eval_dataset)
            #  --------------------------------------------------------------------------------------

            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            image_size = (args.image_size, args.image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
     
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(args.model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            results = verification.ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess,
                               embedding_tensor=embeddings, batch_size=args.batch_size, feed_dict=input_map,
                               input_placeholder=image_batch, phase_train_placeholder=phase_train_placeholder)

            print(results)

            # evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder,
            #          control_placeholder, embeddings, label_batch, paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds,
            #          args.distance_metric, args.subtract_mean, args.use_flipped_images, args.use_fixed_image_standardization)

              
def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder,
             control_placeholder,embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, distance_metric, subtract_mean,
             use_flipped_images,use_fixed_image_standardization):

    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    
    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int32)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
    if use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    
    embedding_size = int(embeddings.get_shape()[1])
    print(f"nrof_images: {nrof_images}")

    batch_size = closest_number(nrof_images, batch_size, 100)

    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)


def closest_number(number, divisible, max_bound):
    for i in range(max_bound+1):
        if number % (divisible + i) == 0:
            return divisible + i
        if number % (divisible - i) == 0:
            return divisible - i

    return None

if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))
    obj_args = Args()
    main(obj_args)
