from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def convert_box(bboxes):
    x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    y = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    x = tf.reshape(x, (tf.size(x), 1))
    y = tf.reshape(y, (tf.size(y), 1))
    w = tf.reshape(w, (tf.size(w), 1))
    h = tf.reshape(h, (tf.size(h), 1))
    return x, y, w, h


def preprocess_bboxes(labels, bboxes, box_num):
    convert_box(bboxes)
    x, y, w, h = convert_box(bboxes)
    bboxes = tf.concat([x, y, w, h, tf.cast(tf.reshape(labels, (tf.size(labels), 1)), tf.float32)], 1)
    boxes = tf.zeros((box_num - tf.shape(bboxes)[0],5), tf.float32)
    boxes = tf.concat([bboxes,boxes],0)
    boxes = tf.reshape(boxes,(box_num,5))

    return boxes


def preprocess_for_train(image, labels, bboxes, out_size, box_num, angle, saturation, exposure, hue, jitter):
    resized_image = tf.image.resize_images(image, out_size)
    bboxes = preprocess_bboxes(labels, bboxes, box_num)

    return resized_image, bboxes


def preprocess_for_eval(image, labels, bboxes, out_size, box_num):
    resized_image = tf.image.resize_images(image, out_size)
    bboxes = preprocess_bboxes(labels, bboxes, box_num)

    return resized_image, bboxes


def preprocess_data(image, labels, bboxes, out_size, box_num, is_training=True, angle=0,
                    saturation=0, exposure=0, hue=0,
                    jitter=0):
    """Preprocesses the given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, then this value
        is used for rescaling.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, this value is
        ignored. Otherwise, the resize side is sampled from
          [resize_size_min, resize_size_max].

    Returns:
      A preprocessed image.
    """
    if is_training:
        return preprocess_for_train(image, labels, bboxes, out_size, box_num, angle, saturation,
                                    exposure, hue, jitter)
    else:
        return preprocess_for_eval(image, labels, bboxes, out_size, box_num)
