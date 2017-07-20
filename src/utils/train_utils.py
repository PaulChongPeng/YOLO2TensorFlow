import tensorflow as tf
slim = tf.contrib.slim


def configure_learning_rate(flags,num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / flags.batch_size *
                    flags.num_epochs_per_decay)
  if flags.sync_replicas:
    decay_steps /= flags.replicas_to_aggregate

  if flags.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(flags.learning_rate,
                                      global_step,
                                      decay_steps,
                                      flags.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif flags.learning_rate_decay_type == 'fixed':
    return tf.constant(flags.learning_rate, name='fixed_learning_rate')
  elif flags.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(flags.learning_rate,
                                     global_step,
                                     decay_steps,
                                     flags.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     flags.learning_rate_decay_type)


def configure_optimizer(flags,learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if flags.optimizer is not recognized.
  """
  if flags.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=flags.adadelta_rho,
        epsilon=flags.opt_epsilon)
  elif flags.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=flags.adagrad_initial_accumulator_value)
  elif flags.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=flags.adam_beta1,
        beta2=flags.adam_beta2,
        epsilon=flags.opt_epsilon)
  elif flags.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=flags.ftrl_learning_rate_power,
        initial_accumulator_value=flags.ftrl_initial_accumulator_value,
        l1_regularization_strength=flags.ftrl_l1,
        l2_regularization_strength=flags.ftrl_l2)
  elif flags.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=flags.momentum,
        name='Momentum')
  elif flags.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=flags.rmsprop_decay,
        momentum=flags.momentum,
        epsilon=flags.opt_epsilon)
  elif flags.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', flags.optimizer)
  return optimizer

def get_init_fn(flags):
  """Returns a function run by the chief worker to warm-start the training.

  Note that the init_fn is only run when initializing the model during the very
  first global step.

  Returns:
    An init function run by the supervisor.
  """
  if flags.checkpoint_path is None:
    return None

  # Warn the user if a checkpoint exists in the train_dir. Then we'll be
  # ignoring the checkpoint anyway.
  if tf.train.latest_checkpoint(flags.train_dir):
    tf.logging.info(
        'Ignoring --checkpoint_path because a checkpoint already exists in %s'
        % flags.train_dir)
    return None

  exclusions = []
  if flags.checkpoint_exclude_scopes:
    exclusions = [scope.strip()
                  for scope in flags.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
  variables_to_restore = []
  for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      variables_to_restore.append(var)

  if tf.gfile.IsDirectory(flags.checkpoint_path):
    checkpoint_path = tf.train.latest_checkpoint(flags.checkpoint_path)
  else:
    checkpoint_path = flags.checkpoint_path

  tf.logging.info('Fine-tuning from %s' % checkpoint_path)

  return slim.assign_from_checkpoint_fn(
      checkpoint_path,
      variables_to_restore,
      ignore_missing_vars=flags.ignore_missing_vars)


def get_variables_to_train(flags):
  """Returns a list of variables to train.

  Returns:
    A list of variables to train by the optimizer.
  """
  if flags.trainable_scopes is None:
    return tf.trainable_variables()
  else:
    scopes = [scope.strip() for scope in flags.trainable_scopes.split(',')]

  variables_to_train = []
  for scope in scopes:
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    variables_to_train.extend(variables)
  return variables_to_train