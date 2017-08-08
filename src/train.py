'''

python src/train.py \
    --train_dir=/raid/pengchong_data/tfmodel_test/ \
    --dataset_dir=/raid/pengchong_data/Data/VOC/VOCdevkit/TFRecords/2007 \
    --max_number_of_steps=100 \
    --batch_size=2

'''
import os
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory, yolo_v2
from preprocessing import yolo_v2_preprocessing

slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 60,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.0005, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'voc_2007', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'yolo_v2', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_classes', 20, 'The number of classes.')

tf.app.flags.DEFINE_integer(
    'train_image_size', (416, 416), 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', 10000,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def inference_sequential(image_batch):
    network_fn = nets_factory.get_network_fn(
        name=FLAGS.model_name,
        num_classes=FLAGS.num_classes,
        is_training=True,
        weight_decay=FLAGS.weight_decay,
        num_anchors=5)
    net, end_points = network_fn(image_batch)

    box_coordinate, box_confidence, box_class_probs = yolo_v2.yolo_v2_head(net, FLAGS.num_classes,
                                                                           [[1, 2], [1, 3], [2, 1], [3, 1], [1, 1]],
                                                                           True)

    # preds = tf.reduce_max(box_class_probs, 4)
    # preds = tf.one_hot(tf.cast(preds, tf.int32), FLAGS.num_classes)

    # return preds

    return box_coordinate, box_confidence, box_class_probs


# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    with tf.Graph().as_default():
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        global_step = tf.train.create_global_step()

        # Select the dataset.
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
        max_box_num_per_image = dataset_factory.get_box_num_per_image(FLAGS.dataset_name, FLAGS.dataset_split_name)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            common_queue_capacity=20 * FLAGS.batch_size,
            common_queue_min=10 * FLAGS.batch_size)
        # Get input for network: image, labels, bboxes.
        [image, glabels, gbboxes, box_num] = provider.get(['image', 'object/label', 'object/bbox', 'box_num'])

        train_image_size = FLAGS.train_image_size
        image, gbboxes = yolo_v2_preprocessing.preprocess_data(image, glabels, gbboxes, train_image_size,
                                                               max_box_num_per_image)

        image_batch, gbboxes_batch = tf.train.batch(
            [image, gbboxes],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size)

        batch_queue = slim.prefetch_queue.prefetch_queue(
            [image_batch, gbboxes_batch], capacity=2)

        image_batch, gbboxes_batch = batch_queue.dequeue()

        summaries.add(tf.summary.image('batch image', image_batch))

        print(gbboxes_batch)

        box_coordinate, box_confidence, box_class_probs = inference_sequential(image_batch)
        total_loss, confidence_loss, coordinate_loss, category_loss, xy_loss, wh_loss, objects_loss, no_objects_loss = yolo_v2.yolo_v2_loss(
            box_coordinate,
            box_confidence,
            box_class_probs,
            [[1, 2], [1, 3], [2, 1],[3, 1], [1, 1]],
            gbboxes_batch,
            num_classes=FLAGS.num_classes)

        summaries.add(tf.summary.scalar('loss_total', total_loss))
        summaries.add(tf.summary.scalar('loss_confidence', confidence_loss))
        summaries.add(tf.summary.scalar('loss_confidence_object', objects_loss))
        summaries.add(tf.summary.scalar('loss_confidence_no_object', no_objects_loss))
        summaries.add(tf.summary.scalar('loss_coordinate', coordinate_loss))
        summaries.add(tf.summary.scalar('loss_coordinate_xy', xy_loss))
        summaries.add(tf.summary.scalar('loss_coordinate_wh', wh_loss))
        summaries.add(tf.summary.scalar('loss_category', category_loss))

        # optimizer = tf.train.GradientDescentOptimizer(0.01)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)

        train_op = slim.learning.create_train_op(total_loss, optimizer)

        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True

        final_loss = slim.learning.train(train_op,
                                         logdir=FLAGS.train_dir,
                                         summary_op=summary_op,
                                         global_step=global_step,
                                         number_of_steps=FLAGS.max_number_of_steps,
                                         log_every_n_steps=FLAGS.log_every_n_steps,
                                         save_summaries_secs=FLAGS.save_summaries_secs,
                                         save_interval_secs=FLAGS.save_interval_secs,
                                         session_config=sess_config)

    print('Finished training. Last batch loss %f' % final_loss)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()
