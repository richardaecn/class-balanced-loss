"""Read CIFAR-10 data from pickled numpy arrays and writes
TFRecords with imblanced classes.

Generates tf.train.Example protos and writes them to TFRecord files from the
python version of the CIFAR-10 dataset downloaded from
https://www.cs.toronto.edu/~kriz/cifar.html.
https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import random
import pickle

from data_utils import *


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _get_file_names(data_version):
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    if data_version == '10':
        file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 6)]
        # file_names['validation'] = ['data_batch_5']
        file_names['eval'] = ['test_batch']
    else:
        file_names['train'] = ['train']
        file_names['eval'] = ['test']
    return file_names


def get_traindata_list(data_dir, data_version):
    """Returns a dict contain a list of file names for training data."""
    training_data = {}
    for cls_idx in range(int(data_version)):
        training_data[str(cls_idx)] = []
    file_names = _get_file_names(data_version)['train']

    cifar_local = local_folder(data_version)
    input_dir = os.path.join(data_dir, cifar_local)

    for f in file_names:
        data_path = os.path.join(input_dir, f)
        data_dict = read_pickle_from_file(data_path)
        for idx, label in enumerate(data_dict[label_names(data_version)]):
            training_data[str(label)].append(f + '/' + str(idx))

    return training_data


def get_imbalanced_data(training_data, img_num_per_cls):
    """Get a list of imbalanced training data, store it into im_data dict."""
    im_data = {}
    for cls_idx, img_id_list in training_data.items():
        random.shuffle(img_id_list)
        img_num = img_num_per_cls[int(cls_idx)]
        im_data[cls_idx] = img_id_list[:img_num]
    return im_data


def sort_input(im_data, cifar_version):
    """Sort data into batch - idx_list for faster tfrecord writing."""
    data_to_write = []
    for cls_idx, img_id_list in im_data.items():
        data_to_write.extend(img_id_list)

    print('number of training images are {}'.format(len(data_to_write)))
    if cifar_version == '10':
        data_sorted = {
            'data_batch_1': [],
            'data_batch_2': [],
            'data_batch_3': [],
            'data_batch_4': [],
            'data_batch_5': [],
        }
    else:
        data_sorted = {'train': []}

    for img_id in data_to_write:
        image_id = img_id.split('/')
        data_sorted[image_id[0]].append(int(image_id[1]))
    return data_sorted


def convert_to_tfrecord(input_dir, output_file, data_version, im_sorted_data):
    """Converts a file to tfrecord."""
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for folder_name, img_list in im_sorted_data.items():
            # Convert to tf.train.Example and write the to TFRecords.
            print('converting data from {}'.format(folder_name))

            input_path = os.path.join(input_dir, folder_name)
            data_dict = read_pickle_from_file(input_path)
            data = data_dict[b'data']
            labels = data_dict[label_names(data_version)]

            for img_id in img_list:
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': _bytes_feature(data[img_id].tobytes()),
                            'label': _int64_feature(labels[img_id])
                        }))
                record_writer.write(example.SerializeToString())


def convert_to_pickle(input_dir, output_file, data_version, im_sorted_data):
    images = []
    labels_list = []
    for folder_name, img_list in im_sorted_data.items():
        # Convert to tf.train.Example and write the to TFRecords.
        print('converting data from {}'.format(folder_name))

        input_path = os.path.join(input_dir, folder_name)
        data_dict = read_pickle_from_file(input_path)
        data = data_dict[b'data']
        labels = data_dict[label_names(data_version)]

        for img_id in img_list:
            image, label = parse_raw(data[img_id], labels[img_id])
            images.append(image)
            labels_list.append(label)
        print(len(images))
        print(len(labels_list))
    images = tf.stack(images, 0)
    print(len(labels_list))
    labels_list = tf.stack(labels_list, 0)
    # save it
    sess = tf.Session()
    with sess.as_default():
        with open(output_file, 'wb') as f:
            pickle.dump([images.eval(), labels_list.eval()], f)


def parse_raw(image, label):
    # Reshape from [depth * height * width] to [depth, height, width].
    HEIGHT = 32
    WIDTH = 32
    DEPTH = 3
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(label, tf.int32)

    return image, label


def main(imb_factor, args):
    data_ver = args.CIFAR_data_version

    print('=' * 80)
    if data_ver == '10':
        print('cifar-10, imbalance factor = ' + str(1.0 / imb_factor) + '%')
    elif data_ver == '20':
        print('cifar-20, imbalance factor = ' + str(1.0 / imb_factor) + '%')
    elif data_ver == '100':
        print('cifar-100, imbalance factor = ' + str(1.0 / imb_factor) + '%')
    print('=' * 80)

    if args.resample:
        ori_data_dir = '/media/user/2tb/2018FocalLoss/data/cifar-' + data_ver + '-data'
        im_data_dir = '/media/user/2tb/2018FocalLoss/data/cifar-' + data_ver + '-data-im-' + str(imb_factor)
        im_data = read_json(os.path.join(im_data_dir, "train_img_id.json"))
    else:
        ori_data_dir = './data/cifar-' + data_ver + '-data'
        im_data_dir = './data/cifar-' + data_ver + '-data-im-' + str(imb_factor)

        orig_data = get_traindata_list(ori_data_dir, data_ver)
        img_num_list = get_img_num_per_cls(data_ver, imb_factor)
        im_data = get_imbalanced_data(orig_data, img_num_list)

    # write_json(orig_data, os.path.join(ori_data_dir, 'train_img_id.json'))
    # write_json(im_data, os.path.join(im_data_dir, 'train_img_id.json'))

    im_sorted_data = sort_input(im_data, data_ver)

    input_dir = os.path.join(ori_data_dir, local_folder(data_ver))
    if args.resample:
        output_file = os.path.join(im_data_dir, 'train.pickle')
    else:
        output_file = os.path.join(im_data_dir, 'train.tfrecords')
    print('writing data to ' + output_file)

    try:
        os.remove(output_file)
    except OSError:
        pass

    if args.resample:
        convert_to_pickle(input_dir, output_file, data_ver, im_sorted_data)
    else:
        convert_to_tfrecord(input_dir, output_file, data_ver, im_sorted_data)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--CIFAR-data-version',
        type=str,
        default='10',
        help='CIFAR data version, 10, 20, or 100')
    parser.add_argument(
        '--resample',
        action='store_true',
        default=False,
        help='Whether to do resample.')
    args = parser.parse_args()

    # imb_factor_list = [0.005, 0.01, 0.02, 0.05, 0.1]
    imb_factor_list = [0.02]

    for imb_factor in imb_factor_list:
        main(imb_factor, args)
