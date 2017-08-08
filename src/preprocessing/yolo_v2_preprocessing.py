from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import tf_utils

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


def get_index(index_0, index_1, index_2, index_3):
    tf_index = tf.concat(
        [tf.reshape(tf.cast(index_0, tf.int64), [1]),
         tf.reshape(tf.cast(index_1, tf.int64), [1]),
         tf.reshape(tf.cast(index_2, tf.int64), [1]),
         tf.reshape(tf.constant(index_3, tf.int64), [1])], 0)
    tf_index = tf.reshape(tf_index, [1, 4])
    return tf_index


def get_index_and_value_x(index_0, index_1, index_2, box, is_training):
    tf_index = get_index(index_0, index_1, index_2, 0)
    if is_training:
        tf_value = tf.reshape(box[0] - index_0, [1])
    else:
        tf_value = tf.reshape(box[0], [1])
    return tf_index, tf_value


def get_index_and_value_y(index_0, index_1, index_2, box, is_training):
    tf_index = get_index(index_0, index_1, index_2, 1)
    if is_training:
        tf_value = tf.reshape(box[1] - index_1, [1])
    else:
        tf_value = tf.reshape(box[1], [1])
    return tf_index, tf_value


def get_index_and_value_w(index_0, index_1, index_2, box, anchor):
    tf_index = get_index(index_0, index_1, index_2, 2)
    tf_value = tf.reshape(box[2], [1])
    #tf_value = tf.reshape(tf.log(box[2] / anchor[0]), [1])
    return tf_index, tf_value


def get_index_and_value_h(index_0, index_1, index_2, box, anchor):
    tf_index = get_index(index_0, index_1, index_2, 3)
    tf_value = tf.reshape(box[3], [1])
    #tf_value = tf.reshape(tf.log(box[3] / anchor[1]), [1])
    return tf_index, tf_value


def get_index_and_value_c(index_0, index_1, index_2, box):
    tf_index = get_index(index_0, index_1, index_2, 4)
    tf_value = tf.reshape(box[4], [1])
    return tf_index, tf_value


def process_gbboxes_with_anchors(gbboxes, image_size, anchors, box_num, is_training):
    gbboxes_coor = gbboxes[:, 0:4] * image_size[0] / 32
    gbboxes = tf.concat([gbboxes_coor, tf.expand_dims(gbboxes[:, 4], 1)], 1)

    indices = []
    values = []

    index = tf.floor(gbboxes[:, 0:2])

    for i in range(box_num):
        box = gbboxes[i]
        max_iou = tf.constant(0, tf.float32)
        index_2 = 0
        anchor_wh = tf.constant([0,0],tf.float32)
        for j, anchor in enumerate(anchors):
            iou = tf_utils.tf_anchor_iou(box, anchor)
            max_iou, index_2, anchor_wh = tf.cond(iou > max_iou, lambda: (iou, j, tf.constant(anchor,tf.float32)),
                                                  lambda: (max_iou, index_2, anchor_wh))

        index_0 = index[i, 0]
        index_1 = index[i, 1]

        tf_index, tf_value = get_index_and_value_x(index_0, index_1, index_2, box, is_training)
        indices.append(tf_index)
        values.append(tf_value)

        tf_index, tf_value = get_index_and_value_y(index_0, index_1, index_2, box, is_training)
        indices.append(tf_index)
        values.append(tf_value)

        tf_index, tf_value = get_index_and_value_w(index_0, index_1, index_2, box, anchor_wh)
        indices.append(tf_index)
        values.append(tf_value)

        tf_index, tf_value = get_index_and_value_h(index_0, index_1, index_2, box, anchor_wh)
        indices.append(tf_index)
        values.append(tf_value)

        tf_index, tf_value = get_index_and_value_c(index_0, index_1, index_2, box)
        indices.append(tf_index)
        values.append(tf_value)

    for temp_index in range(len(indices)):
        if temp_index == 0:
            tf_indices = indices[temp_index]
        else:
            tf_indices = tf.concat([tf_indices, indices[temp_index]], 0)

    print(tf_indices)

    for temp_index in range(len(values)):
        if temp_index == 0:
            tf_values = values[temp_index]
        else:
            tf_values = tf.concat([tf_values, values[temp_index]], 0)

    print(tf_values)

    boxes = tf.SparseTensor(tf_indices, tf_values, [image_size[0] // 32, image_size[1] // 32, len(anchors), 5])

    boxes = tf.sparse_tensor_to_dense(boxes, validate_indices=False)
    # return boxes, tf_indices, tf_values
    return boxes


def preprocess_bboxes(labels, bboxes, box_num, is_training):
    convert_box(bboxes)
    x, y, w, h = convert_box(bboxes)
    bboxes = tf.concat([x, y, w, h, tf.cast(tf.reshape(labels, (tf.size(labels), 1)), tf.float32)], 1)
    boxes = tf.zeros((box_num - tf.shape(bboxes)[0], 5), tf.float32)
    boxes = tf.concat([bboxes, boxes], 0)
    boxes = tf.reshape(boxes, (box_num, 5))

    boxes = process_gbboxes_with_anchors(boxes, [416, 416], [[1, 2], [1, 3], [2, 1], [3, 1], [1, 1]], box_num, is_training)

    return boxes


def preprocess_for_train(image, labels, bboxes, out_size, box_num, angle, saturation, exposure, hue, jitter):
    resized_image = tf.image.resize_images(image, out_size)
    bboxes = preprocess_bboxes(labels, bboxes, box_num, True)

    return resized_image, bboxes


def preprocess_for_eval(image, labels, bboxes, out_size, box_num):
    resized_image = tf.image.resize_images(image, out_size)

    bboxes = preprocess_bboxes(labels, bboxes, box_num, False)


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
