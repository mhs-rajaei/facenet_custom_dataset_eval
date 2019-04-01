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
import os


PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

log_path = os.path.join(PROJECT_PATH, 'output')
models_path = os.path.join(PROJECT_PATH, 'models')


from importlib.machinery import SourceFileLoader
facenet = SourceFileLoader('facenet', os.path.join(PROJECT_PATH, 'MR_facenet.py')).load_module()

eval_data_reader = SourceFileLoader('eval_data_reader', os.path.join(PROJECT_PATH, 'eval_data_reader.py')).load_module()
verification = SourceFileLoader('verification', os.path.join(PROJECT_PATH, 'verification.py')).load_module()
lfw = SourceFileLoader('lfw', os.path.join(PROJECT_PATH, 'lfw.py')).load_module()


def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            # Read the file containing the pairs used for testing
            # InsightFace_TF evaluation
            ver_list = []
            ver_name_list = []
            print('Begin evaluation of %s .' % args.eval_dataset)

            data_set = eval_data_reader.load_eval_datasets_2(args)
            ver_list.append(data_set)
            ver_name_list.append(args.eval_dataset)

            #  --------------------------------------------------------------------------------------

            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')


            image_batch = tf.placeholder(name='img_inputs', shape=[None, args.image_size, args.image_size, 3], dtype=tf.float32)
            label_batch = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int32)

            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(args.model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            results = verification.ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess, embedding_tensor=embeddings,
                                            batch_size=args.batch_size, feed_dict=input_map,input_placeholder=image_batch,
                                            phase_train_placeholder=phase_train_placeholder)

            print(results)


class Args:
    image_size = 160
    batch_size = 32

    eval_pair = os.path.join(PROJECT_PATH, 'data/First_100_ALL VIS_160_1.txt')
    eval_dataset = r"E:\Projects & Courses\CpAE\NIR-VIS-2.0 Dataset -cbsr.ia.ac.cn\First_100_ALL VIS_160"

    lfw_dir = eval_dataset
    lfw_batch_size = batch_size
    model = os.path.join(PROJECT_PATH, 'models/facenet/20180402-114759')
    lfw_pairs = eval_pair


if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))
    obj_args = Args()
    main(obj_args)
