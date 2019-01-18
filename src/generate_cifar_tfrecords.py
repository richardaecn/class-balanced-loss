"""Read CIFAR-10 data from pickled numpy arrays and writes TFRecords.

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
import sys

import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


def download_and_extract(data_dir, data_version):
  print('=' * 80)
  if data_version == '10':
    print('cifar-10 dataset')
    CIFAR_FILENAME = 'cifar-10-python.tar.gz'
  else:
    print('cifar-100 dataset')
    CIFAR_FILENAME = 'cifar-100-python.tar.gz'
  print('=' * 80)

  CIFAR_DOWNLOAD_URL = 'https://www.cs.toronto.edu/~kriz/' + CIFAR_FILENAME

  print('Download from {} and extract.'.format(CIFAR_DOWNLOAD_URL))
  # download CIFAR-10 if not already downloaded.
  tf.contrib.learn.datasets.base.maybe_download(CIFAR_FILENAME, data_dir,
                                                CIFAR_DOWNLOAD_URL)
  tarfile.open(os.path.join(data_dir, CIFAR_FILENAME),
               'r:gz').extractall(data_dir)


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


def read_pickle_from_file(filename):
  with tf.gfile.Open(filename, 'rb') as f:
    if sys.version_info >= (3, 0):
      data_dict = pickle.load(f, encoding='bytes')
    else:
      data_dict = pickle.load(f)
  return data_dict


def convert_to_tfrecord(input_files, output_file, data_version):
  """Converts a file to TFRecords."""
  print('Generating %s' % output_file)
  with tf.python_io.TFRecordWriter(output_file) as record_writer:
    for input_file in input_files:
      data_dict = read_pickle_from_file(input_file)
      data = data_dict[b'data']
      if data_version == '10':
        labels = data_dict[b'labels']
      elif data_version == '100': # cifar-100
        labels = data_dict[b'fine_labels']
      else: # cifar-20
        labels = data_dict[b'coarse_labels']

      num_entries_in_batch = len(labels)
      for i in range(num_entries_in_batch):
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': _bytes_feature(data[i].tobytes()),
                'label': _int64_feature(labels[i])
            }))
        record_writer.write(example.SerializeToString())


def main(args):
  data_dir = args.data_dir
  data_ver = args.CIFAR_data_version

  download_and_extract(data_dir, data_ver)
  file_names = _get_file_names(data_ver)

  if data_ver == '10':
    CIFAR_LOCAL_FOLDER = 'cifar-10-batches-py'
  else:
    CIFAR_LOCAL_FOLDER = 'cifar-100-python'
  input_dir = os.path.join(data_dir, CIFAR_LOCAL_FOLDER)

  for mode, files in file_names.items():
    input_files = [os.path.join(input_dir, f) for f in files]
    output_file = os.path.join(data_dir, mode + '.tfrecords')
    try:
      os.remove(output_file)
    except OSError:
      pass
    # Convert to tf.train.Example and write the to TFRecords.
    convert_to_tfrecord(input_files, output_file, data_ver)
  print('Done!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      default='',
      help='Directory to download and extract CIFAR-10/100 to.')
  parser.add_argument(
      '--CIFAR-data-version',
      type=str,
      default='10',
      help='CIFAR data version, 10, 20, or 100')

  args = parser.parse_args()

  if args.CIFAR_data_version not in ['10', '20', '100']:
    raise ValueError('--CIFAR-data-version: must be one of 10, 20, and 100')

  main(args)
