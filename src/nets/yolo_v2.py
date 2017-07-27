# coding=utf-8
import tensorflow as tf
from utils import tf_utils

slim = tf.contrib.slim


def yolo_v2_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2]):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[3, 3],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
            return arg_sc


def yolo_v2(inputs, num_classes, is_training, num_anchors=5, scope='yolo_v2'):
    with tf.variable_scope(scope, 'yolo_v2', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.conv2d(inputs, 32, scope='layer_0')
            net = slim.max_pool2d(net, scope='layer_1')
            net = slim.conv2d(net, 64, scope='layer_2')
            net = slim.max_pool2d(net, scope='layer_3')
            net = slim.conv2d(net, 128, scope='layer_4')
            net = slim.conv2d(net, 64, kernel_size=[1, 1], scope='layer_5')
            net = slim.conv2d(net, 128, scope='layer_6')
            net = slim.max_pool2d(net, scope='layer_7')
            net = slim.conv2d(net, 256, scope='layer_8')
            net = slim.conv2d(net, 128, kernel_size=[1, 1], scope='layer_9')
            net = slim.conv2d(net, 256, scope='layer_10')
            net = slim.max_pool2d(net, scope='layer_11')
            net = slim.conv2d(net, 512, scope='layer_12')
            net = slim.conv2d(net, 256, kernel_size=[1, 1], scope='layer_13')
            net = slim.conv2d(net, 512, scope='layer_14')
            net = slim.conv2d(net, 256, kernel_size=[1, 1], scope='layer_15')
            net = slim.conv2d(net, 512, scope='layer_16')
            path_1 = tf.space_to_depth(net, block_size=2, name='path_1')
            net = slim.max_pool2d(net, scope='layer_17')
            net = slim.conv2d(net, 1024, scope='layer_18')
            net = slim.conv2d(net, 512, kernel_size=[1, 1], scope='layer_19')
            net = slim.conv2d(net, 1024, scope='layer_20')
            net = slim.conv2d(net, 512, kernel_size=[1, 1], scope='layer_21')
            net = slim.conv2d(net, 1024, scope='layer_22')
            net = slim.conv2d(net, 1024, scope='layer_23')
            net = slim.conv2d(net, 1024, scope='layer_24')
            path_2 = net
            net = tf.concat([path_1, path_2], 3, name='concat2path')
            net = slim.conv2d(net, 1024, scope='layer_25')
            net = slim.conv2d(net, (num_classes + 5) * num_anchors, kernel_size=[1, 1], scope='layer_26')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return net, end_points


def yolo_v2_head(inputs, num_classes, anchors, is_training=True):
    input_shape = tf.shape(inputs)
    anchors_num = len(anchors)

    preds = tf.reshape(inputs, (input_shape[0], input_shape[1], input_shape[2], anchors_num, num_classes + 5))
    box_coordinate = preds[:, :, :, :, 0:4]

    if is_training:
        box_coordinate_xy = tf.sigmoid(box_coordinate[:, :, :, :, 0:2])
        # box_coordinate_wh = box_coordinate[:, :, :, :, 2:4]
        # box_coordinate = tf.concat([box_coordinate_xy, box_coordinate_wh], 4)

        anchors_tensor_w = tf.constant(anchors, tf.float32)[:, 0]
        anchors_tensor_h = tf.constant(anchors, tf.float32)[:, 1]
        anchors_tensor_w = tf.tile(anchors_tensor_w, [input_shape[0] * input_shape[1] * input_shape[2]])
        anchors_tensor_h = tf.tile(anchors_tensor_h, [input_shape[0] * input_shape[1] * input_shape[2]])
        anchors_tensor_w = tf.reshape(anchors_tensor_w, (input_shape[0], input_shape[1], input_shape[2], anchors_num))
        anchors_tensor_h = tf.reshape(anchors_tensor_h, (input_shape[0], input_shape[1], input_shape[2], anchors_num))

        box_coordinate_w = tf.expand_dims(tf.exp(preds[:, :, :, :, 2]) * anchors_tensor_w, 4)
        box_coordinate_h = tf.expand_dims(tf.exp(preds[:, :, :, :, 3]) * anchors_tensor_h, 4)

        box_coordinate = tf.concat([box_coordinate_xy, box_coordinate_w, box_coordinate_h], 4)
    else:
        anchors_tensor_w = tf.constant(anchors, tf.float32)[:, 0]
        anchors_tensor_h = tf.constant(anchors, tf.float32)[:, 1]
        anchors_tensor_w = tf.tile(anchors_tensor_w, [input_shape[0] * input_shape[1] * input_shape[2]])
        anchors_tensor_h = tf.tile(anchors_tensor_h, [input_shape[0] * input_shape[1] * input_shape[2]])
        anchors_tensor_w = tf.reshape(anchors_tensor_w, (input_shape[0], input_shape[1], input_shape[2], anchors_num))
        anchors_tensor_h = tf.reshape(anchors_tensor_h, (input_shape[0], input_shape[1], input_shape[2], anchors_num))

        conv_height_index = tf.range(input_shape[1])
        conv_width_index = tf.range(input_shape[2])
        conv_height_index = tf.tile(conv_height_index, [input_shape[0] * input_shape[2] * anchors_num])
        conv_width_index = tf.tile(conv_width_index, [input_shape[0] * input_shape[1] * anchors_num])
        conv_height_index = tf.reshape(conv_height_index, (input_shape[0], input_shape[1], input_shape[2], anchors_num))
        conv_width_index = tf.reshape(conv_width_index, (input_shape[0], input_shape[1], input_shape[2], anchors_num))

        box_coordinate_x = tf.expand_dims(tf.sigmoid(preds[:, :, :, :, 0]) + tf.cast(conv_width_index, tf.float32), 4)
        box_coordinate_y = tf.expand_dims(tf.sigmoid(preds[:, :, :, :, 1]) + tf.cast(conv_height_index, tf.float32), 4)
        box_coordinate_w = tf.expand_dims(tf.exp(preds[:, :, :, :, 2]) * anchors_tensor_w, 4)
        box_coordinate_h = tf.expand_dims(tf.exp(preds[:, :, :, :, 3]) * anchors_tensor_h, 4)
        box_coordinate = tf.concat([box_coordinate_x, box_coordinate_y, box_coordinate_w, box_coordinate_h], 4)

    box_confidence = preds[:, :, :, :, 4]
    box_class_probs = preds[:, :, :, :, 5:]

    return box_coordinate, box_confidence, box_class_probs


def yolo_v2_confidence_loss(box_coordinate, box_confidence, gbboxes_batch, object_mask, object_scale, no_object_scale):
    iou = tf_utils.tf_boxes_iou(box_coordinate, gbboxes_batch)
    object_no_detections = tf.cast(iou < 0.6, tf.float32)

    no_objects_loss = no_object_scale * (1 - object_mask) * tf.square(0 - box_confidence)
    # 该栅格被标记有物体，但是预测值和标记值的IOU小于0.6,则该栅格的预测值计算object_loss
    objects_loss = object_scale * object_mask * object_no_detections * tf.square(1 - box_confidence)

    no_objects_loss = tf.reduce_sum(no_objects_loss)
    objects_loss = tf.reduce_sum(objects_loss)

    confidence_loss = objects_loss + no_objects_loss
    return confidence_loss, objects_loss, no_objects_loss


def yolo_v2_coordinate_loss(box_coordinate, gbboxes_batch, object_mask, coordinates_scale):
    xy_loss = box_coordinate[..., 0:2] - gbboxes_batch[..., 0:2]
    xy_loss = tf.square(xy_loss)
    xy_loss = object_mask * tf.reduce_sum(xy_loss, 4)
    xy_loss = coordinates_scale * tf.reduce_sum(xy_loss)

    wh_loss = tf.sqrt(box_coordinate[..., 2:4]) - tf.sqrt(gbboxes_batch[..., 2:4])
    wh_loss = tf.square(wh_loss)
    wh_loss = object_mask * tf.reduce_sum(wh_loss, 4)
    wh_loss = coordinates_scale * tf.reduce_sum(wh_loss)

    coordinate_loss = xy_loss + wh_loss
    # coordinate_loss = object_mask * tf.reduce_sum(coordinate_loss, 4)
    # coordinate_loss = coordinates_scale * tf.reduce_sum(coordinate_loss)


    return coordinate_loss, xy_loss, wh_loss


def yolo_v2_category_loss(box_class_probs, gbboxes_batch, object_mask, num_classes, class_scale):
    # TODO 改善为标记的默认box[0,0,0,0,0]对分类损失的贡献
    # 方向：
    # 1.加上背景类
    # 2.默认box的类别one hot 全为0
    gbboxes_classs = tf.cast(gbboxes_batch[..., 4], tf.int32)
    gbboxes_classs = tf.one_hot(gbboxes_classs, num_classes)
    category_loss = gbboxes_classs - box_class_probs
    category_loss = tf.square(category_loss)
    category_loss = object_mask * tf.reduce_sum(category_loss, 4)
    category_loss = class_scale * tf.reduce_sum(category_loss)
    return category_loss


def yolo_v2_loss(box_coordinate, box_confidence, box_class_probs, anchors, gbboxes_batch, num_classes, object_scale=5,
                 no_object_scale=1, class_scale=1, coordinates_scale=1):
    object_mask = tf.reduce_sum(gbboxes_batch, 4)
    object_mask = tf.cast(object_mask > 0, tf.float32)

    confidence_loss, objects_loss, no_objects_loss = yolo_v2_confidence_loss(box_coordinate, box_confidence,
                                                                             gbboxes_batch, object_mask, object_scale,
                                                                             no_object_scale)
    coordinate_loss, xy_loss, wh_loss = yolo_v2_coordinate_loss(box_coordinate, gbboxes_batch, object_mask,
                                                                coordinates_scale)
    category_loss = yolo_v2_category_loss(box_class_probs, gbboxes_batch, object_mask, num_classes, class_scale)

    total_loss = confidence_loss + coordinate_loss + category_loss
    return total_loss, confidence_loss, coordinate_loss, category_loss, xy_loss, wh_loss, objects_loss, no_objects_loss
