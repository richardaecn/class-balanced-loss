import numpy
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

with tf.Graph().as_default():
  batch_size = 100
  data = ['a'] * 9000 + ['b'] * 1000
  labels = [1] * 9000 + [0] * 1000
  data_tensor = ops.convert_to_tensor(data, dtype=dtypes.string)
  label_tensor = ops.convert_to_tensor(labels, dtype=dtypes.int32)
  shuffled_data, shuffled_labels = tf.train.slice_input_producer(
      [data_tensor, label_tensor], shuffle=True, capacity=3 * batch_size)
  target_probs = numpy.array([0.5, 0.5])
  data_batch, label_batch = tf.contrib.training.stratified_sample(
      [shuffled_data], shuffled_labels, target_probs, batch_size,
      queue_capacity=2 * batch_size)

  with tf.Session() as session:
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()
    coordinator = tf.train.Coordinator()
    tf.train.start_queue_runners(session, coord=coordinator)
    num_iter = 500
    sum_ones = 0.
    for idx in range(num_iter):
      if idx % 100 == 0:
        print(idx)
      d, l = session.run([data_batch, label_batch])
      count_ones = l.sum()
      sum_ones += float(count_ones)
      # print('\tpercentage "a" = %.3f' % (float(count_ones) / len(l)))
    print('Overall: {}'.format(sum_ones / (num_iter * batch_size)))
    coordinator.request_stop()
    coordinator.join()
