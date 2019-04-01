import tensorflow as tf
import numpy as np
import pickle
import argparse
import os
import mxnet as mx
import cv2
import io
import PIL.Image
import mxnet.ndarray as nd
from pprint import pprint
from scipy import misc


def get_parser():
    parser = argparse.ArgumentParser(description='evluation data parser')
    parser.add_argument('--eval_dataset', default=['lfw', 'cfp_ff', 'cfp_fp', 'agedb_30'], help='evluation datasets')
    # parser.add_argument('--eval_dataset', default=['cfp_fp'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='../datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--tfrecords_file_path', default='../datasets/tfrecords/eval', help='the image size')
    parser.add_argument('--db_base_path', default='../datasets/faces_ms1m_112x112', help='the image size')
    args = parser.parse_args()
    return args


def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def remove_duplicates(list):
    final_list = []
    for num in list:
        if num not in final_list:
            final_list.append(num)
    return final_list


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            # print(line)
            # pprint(pair)
            if pair not in pairs:
                pairs.append(pair)
    # pairs = remove_duplicates(pairs)
    return np.array(pairs)
    # return pairs


def get_paths(args):
    # args.eval_dataset = os.path.expanduser(args.eval_dataset)
    pairs = read_pairs(os.path.expanduser(args.eval_pair))

    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(args.eval_dataset, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(args.eval_dataset, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(args.eval_dataset, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(args.eval_dataset, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list


def read_pairs_2(pairs_filename):
    from pprint import pprint
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split('\t')
            # print(line)
            # pprint(pair)
            if pair not in pairs:
                pairs.append(pair)
    # pairs = remove_duplicates(pairs)
    return np.array(pairs)


def get_paths_3(args):

    # Read the file containing the pairs used for testing
    pairs = read_pairs_2(os.path.expanduser(args.eval_pair))
    # Get the paths for the corresponding images
    path_list, issame_list = get_paths_2(os.path.expanduser(args.eval_dataset), pairs)

    return path_list, issame_list


def get_paths_2(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = (os.path.join(lfw_dir, pair[0], pair[1]))
            path1 = (os.path.join(lfw_dir, pair[0], pair[2]))

            issame = True

        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[1])
            path1 = os.path.join(lfw_dir, pair[2], pair[3])

            issame = False

        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            print(path0, path1)
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    return path_list, issame_list


def load_bin(path, image_size):
    '''
    :param path: the input file path
    :param image_size: the input image size
    :return: the returned datasets is opencv format BGR  [112, 112, 3]
    '''
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    issame_list_int = list(map(int, issame_list))
    data_list = []
    for _ in [0, 1]:
        data = np.zeros(shape=[len(issame_list)*2, *image_size, 3])
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        tf_images = tf.image.decode_jpeg(_bin)
        tf_images = tf.reshape(tf_images, shape=(112, 112, 3))
        sess = tf.Session()
        images = sess.run(tf_images)
        img_cv = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
        print(np.min(img_cv), np.max(img_cv), img_cv.dtype)
        cv2.imshow('test', img_cv)
        cv2.waitKey(0)
        for flip in [0,1]:
            if flip == 1:
                # print(i, flip)
                img_cv = np.fliplr(img_cv)
                # cv2.imshow('test', img_cv)
                # cv2.waitKey(0)
            data_list[flip][i][:] = img_cv
        i += 1
        if i % 1000 == 0:
            print('loading images', i)
    print(data_list[0].shape)
    return data_list, issame_list


def mx2tfrecords(imgidx, imgrec, args):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    writer = tf.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        encoded_jpg_io = io.BytesIO(img)
        image = PIL.Image.open(encoded_jpg_io)
        np_img = np.array(image)
        img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        img_raw = img.tobytes()
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()


def mx2tfrecords_eval_data(args, db_name):
    '''
    Change evaluation data to tfrecords
    :param args:
    :param type: lfw, ......
    :return:
    '''
    bins, issame_list = pickle.load(open(os.path.join(args.db_base_path, db_name+'.bin'), 'rb'), encoding='bytes')
    output_image_path = os.path.join(args.tfrecords_file_path, db_name+'_eval_data.tfrecords')
    writer_img = tf.python_io.TFRecordWriter(output_image_path)
    for i in range(len(bins)):
        img_info = bins[i]
        img = mx.image.imdecode(img_info).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_b = img.tobytes()
        # # decode test
        # sess = tf.Session()
        # img_2 = tf.decode_raw(img_b, out_type=tf.uint8)
        # img_2 = tf.reshape(img_2, shape=(112, 112, 3))
        # img_2 = tf.image.flip_left_right(img_2)
        # img_2_np = sess.run(img_2)
        # print(img_2_np.shape)
        # cv2.imshow('test', img_2_np)
        # cv2.waitKey(0)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_b]))
        }))
        writer_img.write(example.SerializeToString())  # Serialize To String
        if i % 1000 == 0:
            print('%d num image processed' % i)
    writer_img.close()


def load_bin(db_name, image_size, args):
    bins, issame_list = pickle.load(open(os.path.join(args.eval_db_path, db_name+'.bin'), 'rb'), encoding='bytes')
    data_list = []
    for _ in [0,1]:
        data = np.empty((len(issame_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for flip in [0,1]:
            if flip == 1:
                img = np.fliplr(img)
            data_list[flip][i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list


def load_eval_datasets(args):
    # bins, issame_list = pickle.load(open(os.path.join(args.eval_db_path, db_name+'.bin'), 'rb'), encoding='bytes')
    image_path_list, issame_list = get_paths(args)
    # data_list = []
    data = np.zeros((2, len(issame_list)*2, args.image_size[0], args.image_size[1], 3), dtype=np.float32)
    # for _ in [0,1]:
    #     data = np.zeros((len(issame_list)*2, args.image_size[0], args.image_size[1], 3), dtype=np.float32)
    #     data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = image_path_list[i]
        # img = mx.image.imdecode(_bin).asnumpy()
        img = misc.imread(_bin)
        # img = mx.image.imdecode(img).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for flip in [0, 1]:
            if flip == 1:
                img = np.fliplr(img)
            data[flip][i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading images', i)
    print(data.shape)
    return data, issame_list


def load_eval_datasets_2(args):
    # bins, issame_list = pickle.load(open(os.path.join(args.eval_db_path, db_name+'.bin'), 'rb'), encoding='bytes')
    image_path_list, issame_list = get_paths_3(args)
    # data_list = []
    data = np.zeros((2, len(issame_list)*2, args.image_size, args.image_size, 3), dtype=np.float32)
    # for _ in [0,1]:
    #     data = np.zeros((len(issame_list)*2, args.image_size[0], args.image_size[1], 3), dtype=np.float32)
    #     data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = image_path_list[i]
        # img = mx.image.imdecode(_bin).asnumpy()
        img = misc.imread(_bin)
        # img = mx.image.imdecode(img).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for flip in [0, 1]:
            if flip == 1:
                img = np.fliplr(img)
            data[flip][i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading images', i)
    print(data.shape)
    return data, issame_list

if __name__ == '__main__':
    args = get_parser()
    ver_list = []
    ver_name_list = []
    for db in args.eval_dataset:
        print('begin db %s convert.' % db)
        # mx2tfrecords_eval_data(args, db)
        data_set = load_bin(db, args.image_size)