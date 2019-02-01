"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from data_utils import get_img_num_per_cls, read_pickle_from_file

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class CifarDataSet(object):
    """Cifar data set."""

    def __init__(self,
                 data_dir,
                 data_version='10',
                 subset='train',
                 imb_factor=None,
                 use_distortion=True,
                 resample=False):
        self.data_dir = data_dir
        self.data_version = data_version
        self.subset = subset
        self.imb_factor = imb_factor
        self.use_distortion = use_distortion
        self.resample = resample

    def get_filenames(self):
        if self.subset == 'train_offline':  # so avoid shuffle during make_batch
            return [os.path.join(self.data_dir, 'train' + '.tfrecords')]
        if self.subset in ['train', 'eval']:
            return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of
        # the input format.
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        # Custom preprocessing.
        image = self.preprocess(image)

        return image, label

    def make_resampled_batch(self, batch_size, target_dist):
        """read the images and labels"""
        # convert image / labels into tensors
        filename = os.path.join(self.data_dir, "train.pickle")
        images, labels = read_pickle_from_file(filename)  # [img_num, 32, 32, 3]
        print("done reading the tensors from {}".format(filename))
        data_tensor = ops.convert_to_tensor(images, dtype=dtypes.float32)
        label_tensor = ops.convert_to_tensor(labels, dtype=dtypes.int32)
        label_tensor = tf.expand_dims(label_tensor, 1)
        min_queue_examples = int(
            CifarDataSet.num_examples_per_epoch(
                self.subset, self.imb_factor, self.data_version) * 0.4)
        shuffled_data, shuffled_labels = tf.train.slice_input_producer(
            [data_tensor, label_tensor], shuffle=True,
            capacity=min_queue_examples + 3 * batch_size)

        shuffled_data = tf.reshape(shuffled_data, [-1, HEIGHT * WIDTH * DEPTH])
        image_batch, label_batch = tf.contrib.training.stratified_sample(
            [shuffled_data], shuffled_labels, target_dist, batch_size, enqueue_many=True,
            queue_capacity=2 * batch_size)
        image_batch = image_batch[0]  # returned a list
        image_batch = tf.reshape(image_batch, [-1, HEIGHT, WIDTH, DEPTH])
        # preprocess
        # Pad 4 pixels on each dimension of feature map, done in mini-batch
        image_batch = tf.image.resize_image_with_crop_or_pad(image_batch, 40, 40)
        image_batch = tf.random_crop(image_batch, [batch_size, HEIGHT, WIDTH, DEPTH])
        image_list = []
        for i in range(batch_size):
            image_list.append(tf.image.random_flip_left_right(image_batch[i, :, :, :]))
        image_batch = tf.stack(image_list, 0)
        # image_batch = tf.image.random_flip_left_right(image_batch)
        return image_batch, label_batch
        # # create dataset
        # dataset = tf.data.Dataset.from_tensor_slices((image_batch, label_batch)).repeat()

    def make_batch(self, batch_size):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()

        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames).repeat()
        # Parse records.
        dataset = dataset.map(self.parser, num_parallel_calls = batch_size)

        # Potentially shuffle records.
        if self.subset == 'train':
            min_queue_examples = int(
                CifarDataSet.num_examples_per_epoch(
                    self.subset, self.imb_factor, self.data_version) * 0.4)
            # Ensure that the capacity is sufficiently large to provide good
            # random shuffling.
            dataset = dataset.shuffle(buffer_size = min_queue_examples +
                                      3 * batch_size)
        # Batch it up.
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()

        return image_batch, label_batch

    def preprocess(self, image):
        """Preprocess a single image in [height, width, depth] layout."""
        if self.subset == 'train' and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
            image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
            image = tf.image.random_flip_left_right(image)
        return image

    @staticmethod
    def num_examples_per_epoch(subset = 'train',
                               imb_factor = None,
                               cifar_version = '10'):
        if subset == 'train':
            if imb_factor is None:
                return 50000
            else:
                return sum(get_img_num_per_cls(cifar_version, imb_factor))
        elif subset == 'eval':
            return 10000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)
