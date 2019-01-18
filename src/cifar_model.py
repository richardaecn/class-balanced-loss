"""Model class for Cifar Dataset."""
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import model_base


class ResNetCifar(model_base.ResNet):
  """Cifar model with ResNetV1 and basic residual block."""

  def __init__(self,
               num_layers,
               is_training,
               batch_norm_decay,
               batch_norm_epsilon,
               version='v1',
               num_classes=10,
               data_format='channels_first',
               loss_type='softmax'):
    super(ResNetCifar, self).__init__(
        is_training,
        data_format,
        batch_norm_decay,
        batch_norm_epsilon
    )
    self.n = (num_layers - 2) // 6
    self.num_classes = num_classes
    self.filters = [16, 16, 32, 64]
    self.strides = [1, 2, 2]
    self.version = version
    self.loss_type = loss_type

  def forward_pass(self, x, input_data_format='channels_last'):
    """Build the core model within the graph."""
    if self._data_format != input_data_format:
      if input_data_format == 'channels_last':
        # Computation requires channels_first.
        x = tf.transpose(x, [0, 3, 1, 2])
      else:
        # Computation requires channels_last.
        x = tf.transpose(x, [0, 2, 3, 1])

    # Image standardization.
    x = x / 128 - 1

    x = self._conv(x, 3, 16, 1)
    x = self._batch_norm(x)
    x = self._relu(x)

    if self.version == 'v1':
      # Use basic (non-bottleneck) block and ResNet V1 (post-activation).
      res_func = self._residual_v1
    elif self.version == 'v2':
      # Use basic (non-bottleneck) block and ResNet V2 (pre-activation).
      res_func = self._residual_v2
    else:  # 'bv2'
      # Use bottleneck block and ResNet V2 (pre-activation).
      res_func = self._bottleneck_residual_v2

    # 3 stages of block stacking.
    for i in range(3):
      with tf.name_scope('stage'):
        for j in range(self.n):
          if j == 0:
            # First block in a stage, filters and strides may change.
            x = res_func(x, self.filters[i], self.filters[i + 1],
                         self.strides[i])
          else:
            # Following blocks in a stage, constant filters and unit stride.
            x = res_func(x, self.filters[i + 1], self.filters[i + 1], 1)

    x = self._global_avg_pool(x)
    x = self._fully_connected(x, self.num_classes, self.loss_type)

    return x
