"""Training and evaluation for CIFAR image classification."""
from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import os
import time

import cifar
import cifar_model
import cifar_utils
import data_utils
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def dir2version(data_dir):
  return data_dir.split('/')[-1].split('-')[1]  # string, 10, 20 or 100


def learning_rate_schedule(current_epoch,
                           base_learning_rate,
                           lr_boundaries,
                           lr_multiplier):
  """Handles linear scaling rule, gradual warmup, and LR decay.
  The learning rate starts at 0, then it increases linearly per epoch.
  After 5 epochs we reach the base learning rate.

  Args:
    current_epoch: `Tensor` for current epoch.
    base_learning_rate: initial learning rate after warmup.
    lr_boundaries: a list of training epochs.
    lr_multiplier: a list of learing rate multipliers.
  Returns:
    A scaled `Tensor` for current learning rate.
  """
  staged_lr = [base_learning_rate * x for x in lr_multiplier]
  decay_rate = (base_learning_rate * current_epoch / lr_boundaries[0])
  for st_lr, start_epoch in zip(staged_lr, lr_boundaries):
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, st_lr)
  return decay_rate


def get_model_fn(num_gpus, variable_strategy, num_workers):
  """Returns a function that will build the resnet model."""

  def _resnet_model_fn(features, labels, mode, params):
    """Resnet model body.

    Support single host, one or more GPU training. Parameter distribution can
    be either one of the following scheme.
    1. CPU is the parameter server and manages gradient updates.
    2. Parameters are distributed evenly across all GPUs, and the first GPU
       manages gradient updates.

    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    weight_decay = params.weight_decay
    momentum = params.momentum
    data_version = params.data_version
    imb_factor = params.imb_factor

    if num_gpus == 0:
      num_devices = 1
      device_type = 'cpu'
    else:
      num_devices = num_gpus
      device_type = 'gpu'

    tower_features = features
    # for estimator mode =PREDICT
    tower_labels = labels if labels is not None else [labels] * num_devices
    tower_losses = []
    reg_losses = []
    tower_gradvars = []
    tower_preds = []
    # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
    # on CPU. The exception is Intel MKL on CPU which is optimal with
    # channels_last.
    data_format = params.data_format
    if not data_format:
      if num_gpus == 0:
        data_format = 'channels_last'
      else:
        data_format = 'channels_first'

    for i in range(num_devices):
      worker_device = '/{}:{}'.format(device_type, i)
      if variable_strategy == 'CPU':
        device_setter = cifar_utils.local_device_setter(
            worker_device=worker_device)
      elif variable_strategy == 'GPU':
        device_setter = cifar_utils.local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
      with tf.variable_scope('resnet', reuse=bool(i != 0)):
        with tf.name_scope('tower_%d' % i) as name_scope:
          with tf.device(device_setter):
            loss_list, gradvars, preds, one_hot_labels = _tower_fn(
                is_training, weight_decay, tower_features[i], tower_labels[i],
                data_version, data_format, params.num_layers,
                params.batch_norm_decay, params.batch_norm_epsilon,
                params.resnet_version, params.loss_type, params.gamma,
                params.weights)
            if mode != tf.estimator.ModeKeys.PREDICT:
              tower_losses.append(loss_list[0])
              reg_losses.append(loss_list[1])
              tower_gradvars.append(gradvars)
            tower_preds.append(preds)
            if i == 0:
              # Only trigger batch_norm moving mean and variance update from
              # the 1st tower. Ideally, we should grab the updates from all
              # towers but these stats accumulate extremely fast so we can
              # ignore the other stats from the other towers without
              # significant detriment.
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                             name_scope)

    if mode != tf.estimator.ModeKeys.PREDICT:
      # Now compute global loss and gradients.
      gradvars = []
      with tf.name_scope('gradient_averaging'):
        all_grads = {}
        for grad, var in itertools.chain(*tower_gradvars):
          if grad is not None:
            all_grads.setdefault(var, []).append(grad)
        for var, grads in six.iteritems(all_grads):
          # Average gradients on the same device as the variables
          # to which they apply.
          with tf.device(var.device):
            if len(grads) == 1:
              avg_grad = grads[0]
            else:
              avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
          gradvars.append((avg_grad, var))

      # Device that runs the ops to apply global gradient updates.
      consolid_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
      with tf.device(consolid_device):
        num_train_examples = cifar.CifarDataSet.num_examples_per_epoch(
            'train', imb_factor, data_version)
        train_batch_size = params.train_batch_size * num_workers
        num_per_batch = train_batch_size / num_train_examples
        current_epoch = tf.cast(
            tf.train.get_global_step(), tf.float32) * num_per_batch
        boundaries = np.asarray(params.learning_rate_schedule, dtype=np.int64)
        multipliers = np.asarray(
            params.learning_rate_multiplier, dtype=np.float32)
        # Linear scaling of base learning rate.
        base_lr = params.learning_rate * train_batch_size / 128
        learning_rate = learning_rate_schedule(
            current_epoch, base_lr, boundaries, multipliers)

        loss = tf.reduce_mean(tower_losses, name='loss')
        reg_loss = tf.reduce_mean(reg_losses, name='regularization_loss')

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum)

        if params.sync:
          optimizer = tf.train.SyncReplicasOptimizer(
              optimizer, replicas_to_aggregate=num_workers)

        # Create single grouped train op
        train_op = [
            optimizer.apply_gradients(
                gradvars, global_step=tf.train.get_global_step())
        ]
        train_op.extend(update_ops)
        train_op = tf.group(*train_op)
    else:
      train_op = None
      loss = None
      # train_hooks = None

    predictions = {
        'classes':
            tf.concat([p['classes'] for p in tower_preds], axis=0),
        'probabilities':
            tf.concat([p['probabilities'] for p in tower_preds], axis=0),
        'logits':
            tf.concat([p['logits'] for p in tower_preds], axis=0),
    }

    if mode != tf.estimator.ModeKeys.PREDICT:
      stacked_labels = tf.concat(labels, axis=0)
      accuracy = tf.metrics.accuracy(stacked_labels, predictions['classes'])

      metrics = {'accuracy': accuracy}

      tf.summary.scalar('learning_rate', learning_rate)
      tf.summary.scalar('regularization_loss', reg_loss)
      tf.summary.scalar('network_loss', loss - reg_loss)
      tf.summary.scalar('epoch', current_epoch)

      one_hot_sum = tf.reduce_sum(one_hot_labels, 0)
      for n in range(int(data_version)):
        tf.summary.scalar('labels/' + str(n), one_hot_sum[n])
    else:
      metrics = {}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        # training_hooks=train_hooks,
        eval_metric_ops=metrics)

  return _resnet_model_fn


def focal_loss(labels, logits, alpha, gamma):
  """Compute the focal loss between `logits` and the ground truth `labels`.

  Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.
  pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

  Args:
    labels: A float32 tensor of size [batch, num_classes].
    logits: A float32 tensor of size [batch, num_classes].
    alpha: A float32 tensor of size [batch_size]
      specifying per-example weight for balanced cross entropy.
    gamma: A float32 scalar modulating loss from hard and easy examples.
  Returns:
    focal_loss: A float32 scalar representing normalized total loss.
  """
  with tf.name_scope('focal_loss'):
    logits = tf.cast(logits, dtype=tf.float32)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits)

    # positive_label_mask = tf.equal(labels, 1.0)
    # probs = tf.sigmoid(logits)
    # probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
    # # With gamma < 1, the implementation could produce NaN during back prop.
    # modulator = tf.pow(1.0 - probs_gt, gamma)

    # A numerically stable implementation of modulator.
    if gamma == 0.0:
      modulator = 1.0
    else:
      modulator = tf.exp(-gamma * labels * logits - gamma * tf.log1p(
          tf.exp(-1.0 * logits)))

    loss = modulator * cross_entropy

    weighted_loss = alpha * loss
    focal_loss = tf.reduce_sum(weighted_loss)
    # Normalize by the total number of positive samples.
    focal_loss /= tf.reduce_sum(labels)
  return focal_loss


def _tower_fn(is_training, weight_decay, feature, label, data_version,
              data_format, num_layers, batch_norm_decay, batch_norm_epsilon,
              resnet_version, loss_type, gamma, weights):
  """Build computation tower (Resnet).

  Args:
    is_training: true if is training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    data_version: a str, '10' or '100'
    data_format: channels_last (NHWC) or channels_first (NCHW).
    num_layers: number of layers, an int.
    batch_norm_decay: decay for batch normalization, a float.
    batch_norm_epsilon: epsilon for batch normalization, a float.
    resnet_version: preactivation or postactivation
    loss_type: Loss type ('softmax', 'sigmoid', 'focal').
    gamma: gamma for focal loss.
    weights: weights per class.

  Returns:
    A tuple with the loss for the tower, the gradients and parameters, and
    predictions.

  """
  num_classes = int(data_version)
  model = cifar_model.ResNetCifar(
      num_layers,
      batch_norm_decay=batch_norm_decay,
      batch_norm_epsilon=batch_norm_epsilon,
      is_training=is_training,
      version=resnet_version,
      num_classes=num_classes,
      data_format=data_format,
      loss_type=loss_type)
  logits = model.forward_pass(feature, input_data_format='channels_last')
  if loss_type == 'softmax':
    tower_pred = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,  # a tensor,
        'labels': label,
    }
  elif loss_type == 'sigmoid' or loss_type == 'focal':
    tower_pred = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.sigmoid(logits),
        'logits': logits,  # a tensor,
        'labels': label,
    }

  if label is None:   # for classifier.predict
    return None, None, tower_pred

  one_hot_labels = tf.one_hot(label, num_classes)

  weights = tf.cast(weights, dtype=tf.float32)
  weights = tf.expand_dims(weights, 0)
  weights = tf.tile(weights, [tf.shape(one_hot_labels)[0], 1]) * one_hot_labels
  weights = tf.reduce_sum(weights, axis=1)
  weights = tf.expand_dims(weights, 1)
  weights = tf.tile(weights, [1, num_classes])

  if loss_type == 'softmax':
    tower_loss = tf.losses.softmax_cross_entropy(
        one_hot_labels, logits, weights=tf.reduce_mean(weights, axis=1))
    tower_loss = tf.reduce_mean(tower_loss)
  elif loss_type == 'sigmoid':
    tower_loss = weights * tf.nn.sigmoid_cross_entropy_with_logits(
        labels=one_hot_labels, logits=logits)
    # Normalize by the total number of positive samples.
    tower_loss = tf.reduce_sum(tower_loss) / tf.reduce_sum(one_hot_labels)
  elif loss_type == 'focal':
    tower_loss = focal_loss(one_hot_labels, logits, weights, gamma)

  model_params = tf.trainable_variables()
  if loss_type == 'softmax':
    reg_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params])
  elif loss_type == 'sigmoid' or loss_type == 'focal':
    # no regularization (weight decay) for last layer's bias.
    reg_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params if 'dense/bias' not in v.name])

  tower_loss += reg_loss
  tower_grad = tf.gradients(tower_loss, model_params)

  return [tower_loss, reg_loss], zip(tower_grad, model_params), tower_pred, one_hot_labels


def input_fn(data_dir,
             subset,
             imbalance_factor,
             num_shards,
             batch_size,
             use_distortion_for_training=True,
             resample=False,
             target_dist=None):
  """Create input graph for model.

  Args:
    data_dir: Directory where TFRecords representing the dataset are located.
    subset: one of 'train', 'validate' and 'eval'.
    imbalance_factor: float, None if this dataset is not long tailed.
    num_shards: num of towers participating in data-parallel training.
    batch_size: total batch size for training to be divided by the number of
    shards.
    use_distortion_for_training: True to use distortions.
    resample: True to use resampling during training.
    target_dist: None is is_resample is false, else is an array of length num_cls.
  Returns:
    two lists of tensors for features and labels, each of num_shards length.
  """
  if resample and target_dist is None:
    raise ValueError(
        'Target distribution should not be None if using resampling')

  with tf.device('/cpu:0'):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = cifar.CifarDataSet(
        data_dir, dir2version(data_dir),
        subset, imbalance_factor, use_distortion, resample)

    if resample:
      image_batch, label_batch = dataset.make_resampled_batch(batch_size, target_dist)
    else:
      image_batch, label_batch = dataset.make_batch(batch_size)

    if num_shards <= 1:
      # No GPU available or only 1 GPU.
      return [image_batch], [label_batch]

    # Note that passing num=batch_size is safe here, even though
    # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
    # examples. This is because it does so only when repeating for a limited
    # number of epochs, but our dataset repeats forever.
    image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
    label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
    feature_shards = [[] for i in range(num_shards)]
    label_shards = [[] for i in range(num_shards)]
    for i in xrange(batch_size):
      idx = i % num_shards
      feature_shards[idx].append(image_batch[i])
      label_shards[idx].append(label_batch[i])
    feature_shards = [tf.parallel_stack(x) for x in feature_shards]
    label_shards = [tf.parallel_stack(x) for x in label_shards]
    return feature_shards, label_shards


def main(job_dir, data_dir, num_gpus, variable_strategy,
         use_distortion_for_training, log_device_placement, num_intra_threads,
         **hparams):
  # The env variable is on deprecation path, default is set to off.
  os.environ['TF_SYNC_ON_FINISH'] = '0'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Session configuration.
  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=log_device_placement,
      intra_op_parallelism_threads=num_intra_threads,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True, allow_growth=True)
  )

  config = cifar_utils.RunConfig(
      session_config=sess_config,
      model_dir=job_dir,
      save_summary_steps=100)

  # Normalized weights based on inverse number of effective data per class.
  img_num_per_cls = data_utils.get_img_num_per_cls(
      hparams['data_version'], hparams['imb_factor'])
  effective_num = 1.0 - np.power(hparams['beta'], img_num_per_cls)
  weights = (1.0 - hparams['beta']) / np.array(effective_num)
  weights = weights / np.sum(weights) * int(hparams['data_version'])

  hparams = tf.contrib.training.HParams(
      is_chief=config.is_chief,
      weights=weights,
      **hparams)

  if hparams.is_resample:
    target_dist = weights / int(hparams.data_version)
  else:
    target_dist = None
  train_input_fn = functools.partial(
      input_fn,
      data_dir,
      subset='train',
      imbalance_factor=hparams.imb_factor,
      num_shards=num_gpus,
      batch_size=hparams.train_batch_size,
      use_distortion_for_training=use_distortion_for_training,
      resample=hparams.is_resample,
      # target_dist=[0.1] * 10,
      target_dist=target_dist,
  )

  eval_input_fn = functools.partial(
      input_fn,
      data_dir,
      subset='eval',
      imbalance_factor=hparams.imb_factor,
      batch_size=hparams.eval_batch_size,
      num_shards=num_gpus)

  num_eval_examples = cifar.CifarDataSet.num_examples_per_epoch('eval')
  if num_eval_examples % hparams.eval_batch_size != 0:
    raise ValueError(
        'validation set size must be multiple of eval_batch_size')

  num_workers = config.num_worker_replicas or 1
  num_train_examples = cifar.CifarDataSet.num_examples_per_epoch(
      'train', hparams.imb_factor, hparams.data_version)
  train_batch_size = hparams.train_batch_size * num_workers
  train_steps = num_train_examples * hparams.train_epochs // train_batch_size
  eval_steps = num_eval_examples // hparams.eval_batch_size
  classifier = tf.estimator.Estimator(
      model_fn=get_model_fn(num_gpus, variable_strategy, num_workers),
      model_dir=job_dir,
      config=config,
      params=hparams)

  ckpt = tf.train.get_checkpoint_state(job_dir)
  if ckpt is None:
    current_step = 0
  else:
    current_step = int(
        os.path.basename(ckpt.model_checkpoint_path).split('-')[1])

  steps_per_eval = num_train_examples * hparams.eval_epochs // train_batch_size
  steps_per_epoch = num_train_examples // train_batch_size
  tf.logging.info('Training for %d steps. Current step %d.',
                  train_steps,
                  current_step)
  start_timestamp = time.time()  # This time will include compilation time

  while current_step < train_steps:
    # Train for up to steps_per_eval number of steps.
    # At the end of training, a checkpoint will be written to --job_dir.
    next_checkpoint = min(current_step + steps_per_eval, train_steps)
    classifier.train(
        input_fn=train_input_fn, max_steps=next_checkpoint)
    current_step = next_checkpoint
    # while current_step < next_checkpoint:
    #   # train for one epoch
    #   # resample first
    #   _next_checkpoint = min(current_step + steps_per_epoch, next_checkpoint)
    #   classifier.train(
    #       input_fn=train_input_fn, max_steps=_next_checkpoint)
    #   current_step = _next_checkpoint

    tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                    next_checkpoint, int(time.time() - start_timestamp))

    # Evaluate the model on the most recent model in --job_dir.
    # Since evaluation happens in batches of --eval_batch_size, some images
    # may be excluded modulo the batch size. As long as the batch size is
    # consistent, the evaluated images are also consistent.
    tf.logging.info('Starting to evaluate.')
    eval_results = classifier.evaluate(
        input_fn=eval_input_fn, steps=eval_steps)
    tf.logging.info('Eval results at step %d: %s',
                    next_checkpoint, eval_results)

  elapsed_time = int(time.time() - start_timestamp)
  tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                  train_steps, elapsed_time)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      required=True,
      help='The directory where the CIFAR-10 input data is stored.')
  parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='The directory where the model will be stored.')
  parser.add_argument(
      '--data-version',
      type=str,
      default='10',
      help='cifar dataset version, 10, 20 or 100')
  parser.add_argument(
      '--variable-strategy',
      choices=['CPU', 'GPU'],
      type=str,
      default='GPU',
      help='Where to locate variable operations')
  parser.add_argument(
      '--num-gpus',
      type=int,
      default=1,
      help='The number of gpus used. Uses only CPU if set to 0.')
  parser.add_argument(
      '--num-layers',
      type=int,
      default=32,
      help='The number of layers of the model.')
  parser.add_argument(
      '--resnet-version',
      type=str,
      default='v1',
      help="""\
      The version of resnet, \
      v1 : use basic (non-bottleneck) block and ResNet V1 (post-activation). \
      v2: Use basic (non-bottleneck) block and ResNet V2 (pre-activation). \
      bv2: Use bottleneck block and ResNet V2 (pre-activation).\
      """)
  parser.add_argument(
      '--train-epochs',
      type=int,
      default=200,
      help='The number of epochs to use for training.')
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=128,
      help='Batch size for training.')
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=100,
      help='Batch size for validation.')
  parser.add_argument(
      '--eval-epochs',
      type=int,
      default=2,
      help='The number of epochs between evaluations.')
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.9,
      help='Momentum for MomentumOptimizer.')
  parser.add_argument(
      '--weight-decay',
      type=float,
      default=2e-4,
      help='Weight decay for convolutions.')
  parser.add_argument(
      '--learning-rate',
      type=float,
      default=0.1,
      help="""\
      This is the inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """)
  parser.add_argument(
      '--learning-rate-schedule',
      nargs='+',
      type=int,
      default=[5, 160, 180],
      help='Schedule of learning rate decay')
  parser.add_argument(
      '--learning-rate-multiplier',
      nargs='+',
      type=float,
      default=[1, 0.1, 0.01],
      help='Schedule of learning rate decay')
  parser.add_argument(
      '--use-distortion-for-training',
      type=bool,
      default=True,
      help='If doing image distortion for training.')
  parser.add_argument(
      '--sync',
      action='store_true',
      default=False,
      help="""\
      If present, running in a distributed environment will run on sync mode.\
      """)
  parser.add_argument(
      '--num-intra-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for intra-op parallelism. When training on CPU
      set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.\
      """)
  parser.add_argument(
      '--num-inter-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """)
  parser.add_argument(
      '--data-format',
      type=str,
      default=None,
      help="""\
      If not set, the data format best for the training device is used.
      Allowed values: channels_first (NCHW) channels_last (NHWC).\
      """)
  parser.add_argument(
      '--log-device-placement',
      action='store_true',
      default=False,
      help='Whether to log device placement.')
  parser.add_argument(
      '--batch-norm-decay',
      type=float,
      default=0.9,
      help='Decay for batch norm.')
  parser.add_argument(
      '--batch-norm-epsilon',
      type=float,
      default=1e-5,
      help='Epsilon for batch norm.')
  parser.add_argument(
      '--imb-factor',
      type=float,
      default=None,
      help='Imbalance factor, None if the dataset is default.')
  parser.add_argument(
      '--loss-type',
      type=str,
      default='softmax',
      help="""\
      Loss type for training the network ('softmax', 'sigmoid', 'focal').\
      """)
  parser.add_argument(
      '--gamma',
      type=float,
      default=1.0,
      help='Gamma for focal loss.')
  parser.add_argument(
      '--beta',
      type=float,
      default=0.0,
      help='Beta for class balanced loss.')
  parser.add_argument(
      '--is-resample',
      action='store_true',
      default=False,
      help='Whether to resample during training.')

  args = parser.parse_args()

  if args.num_gpus > 0:
    assert tf.test.is_gpu_available(), "Requested GPUs but none found."
  if args.num_gpus < 0:
    raise ValueError(
        'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
  if args.num_gpus == 0 and args.variable_strategy == 'GPU':
    raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                     '--variable-strategy=CPU.')
  if (args.num_layers - 2) % 6 != 0:
    raise ValueError('Invalid --num-layers parameter.')
  if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
    raise ValueError('--train-batch-size must be multiple of --num-gpus.')
  if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
    raise ValueError('--eval-batch-size must be multiple of --num-gpus.')
  if args.resnet_version not in ['v1', 'v2', 'bv2']:
    raise ValueError('--resnet-version: must be one of v1, v2, bv2.')
  if args.loss_type not in ['softmax', 'sigmoid', 'focal']:
    raise ValueError('--loss-type must be one of softmax, sigmoid, focal.')
  if len(args.learning_rate_schedule) != len(args.learning_rate_multiplier):
    raise ValueError('The length of --learning-rate-multiplier and '
                     '--learning-rate-schedule must be same.')

  main(**vars(args))
