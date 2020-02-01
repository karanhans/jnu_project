import datetime
import multiprocessing
import os
import re

from distutils.version import LooseVersion

import keras
import keras.backend as K
import keras.engine as KE
import keras.layers as KL
import keras.models as KM
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_applications import get_keras_submodule
from keras_applications.mobilenet import _depthwise_conv_block
from keras_applications.mobilenet_v2 import MobileNetV2
from pytz import timezone

from myolo import visualize
import myolo.myolo_utils as mutils
from myolo.config import Config as config

assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

backend = get_keras_submodule('backend')



def relu6(x):
    return backend.relu(x, max_value=6)


def conv_block(inputs, filters, alpha=1.0, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = KL.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = KL.Conv2D(filters, kernel,
                  padding='valid',
                  use_bias=False,
                  strides=strides,
                  name='conv1')(x)
    x = KL.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return KL.Activation(relu6, name='conv1_relu')(x)


def mobilenet_graph(input_image, architecture, stage5=False, alpha=1.0, depth_multiplier=1):
    
    assert architecture == 'mobilenet'

    x = conv_block(input_image, 32, strides=(2, 2))

    # 112x112x32
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, strides=(2, 2), block_id=2)

    # 56x56x64
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)

    # 28x28x256
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=6)  # added by me

    return x 



def yolo_custom_loss(y_true, y_pred, true_boxes): 
    mask_shape = tf.shape(y_true)[:4] 

    cell_x = tf.to_float(
        tf.reshape(tf.tile(tf.range(config.GRID_W), [config.GRID_H]), (1, config.GRID_H, config.GRID_W, 1, 1)))
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [config.BATCH_SIZE, 1, 1, config.N_BOX, 1])

    coord_mask = tf.zeros(mask_shape)
    conf_mask = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    seen = tf.Variable(0.)
    total_recall = tf.Variable(0.)

    
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(config.ANCHORS, [1, 1, 1, config.N_BOX, 2])

    pred_box_conf = tf.sigmoid(y_pred[..., 4])   

   
    pred_box_class = y_pred[..., 5:]

    true_box_xy = y_true[..., 0:2]  

   
    true_box_wh = y_true[..., 2:4] 

    
    true_wh_half = true_box_wh / 2.
    true_mins = true_box_xy - true_wh_half
    true_maxes = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins = pred_box_xy - pred_wh_half
    pred_maxes = pred_box_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    true_box_conf = iou_scores * y_true[..., 4]      

    
    true_box_class = tf.argmax(y_true[..., 5:], -1) 

     coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * config.COORD_SCALE  # here expand dims has the same effect of reshape

    true_xy = true_boxes[..., 0:2]  
    true_wh = true_boxes[..., 2:4]

    true_wh_half = true_wh / 2.
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half

    intersect_mins = tf.maximum(pred_mins, true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * config.NO_OBJECT_SCALE

   
    conf_mask = conf_mask + y_true[..., 4] * config.OBJECT_SCALE

    class_mask = y_true[..., 4] * tf.gather(config.CLASS_WEIGHTS, true_box_class) * config.CLASS_SCALE

    no_boxes_mask = tf.to_float(coord_mask < config.COORD_SCALE / 2.)
    seen = tf.assign_add(seen, 1.)

    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, config.WARM_UP_BATCHES),
                                                   lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                            true_box_wh + tf.ones_like(true_box_wh) * np.reshape(
                                                                config.ANCHORS,
                                                                [1, 1, 1, config.N_BOX, 2]) * no_boxes_mask,
                                                            tf.ones_like(coord_mask)],
                                                   lambda: [true_box_xy,
                                                            true_box_wh,
                                                            coord_mask])

    
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    loss = loss_xy + loss_wh + loss_conf + loss_class

    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

    
    current_recall = nb_pred_box / (nb_true_box + 1e-6)
    total_recall = tf.assign_add(total_recall, current_recall)

    loss = tf.Print(loss, [loss_xy], message='\nLoss XY \t', summarize=1000)
    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
    loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)

    return loss


def yolo_branch_graph(x, config, alpha=1.0, depth_multiplier=1):
  
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=7)

    # 14x14x512
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=12)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=13)

    # 7x7x1024
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=14)

    # yolo output
    x = KL.Conv2D(config.N_BOX * (4 + 1 + config.NUM_CLASSES), (1, 1), strides=(1, 1),
                  padding='same', name='conv_23')(x)
    output = KL.Reshape((config.GRID_H, config.GRID_W, config.N_BOX, 4 + 1 + config.NUM_CLASSES))(x)

   
    return output


def build_yolo_model(config, depth):
    
    input_feature_map = KL.Input(shape=[None, None, depth], name="input_yolo_feature_map")
   output = yolo_branch_graph(input_feature_map, config)

    return KM.Model([input_feature_map], output, name="yolo_model")



def log2_graph(x):
    return tf.log(x) / tf.log(2.0)


class PyramidROIAlign(KE.Layer):
    

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        boxes = inputs[0]

       
        feature_maps = inputs[1:]

        x1, y1, x2, y2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        
        image_shape = [224, 224]
     
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))
      
        roi_level = tf.minimum(0, tf.maximum(       # constrain roi_level to only 0
            0, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # new_roi_level = tf.cast(5, tf.int8)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(0, 1)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[1][-1],)




def overlaps_graph(boxes1, boxes2):
   
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])

    # 2. Compute intersections
    b1_x1, b1_y1, b1_x2, b1_y2 = tf.split(b1, 4, axis=1)
    b2_x1, b2_y1, b2_x2, b2_y2 = tf.split(b2, 4, axis=1)

    x1 = tf.maximum(b1_x1, b2_x1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x2 = tf.minimum(b1_x2, b2_x2)
    y2 = tf.minimum(b1_y2, b2_y2)

    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])

    return overlaps


def detect_mask_target_graph(yolo_proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    yolo_rois: [7x7x3, (xmin, ymin, xmax, ymax)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [TRUE_BOX_BUFFER] int class IDs
    gt_boxes: [TRUE_BOX_BUFFER, (xmin, ymin, xmax, ymax)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (xmin, ymin, xmax, ymax)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    # deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(yolo_proposals)[0], 0), [yolo_proposals],
                  name='yolo_proposals_assertion'),
    ]
    with tf.control_dependencies(asserts):
        yolo_proposals = tf.identity(yolo_proposals)

    # Remove zero padding
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name='trim_gt_boxes')
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name='trim_gt_class_ids')
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name='trim_gt_masks')

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    # crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    # non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    # crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    # crowd_masks = tf.gather(gt_masks, crowd_ix, axis=2)
    # gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    # gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    # gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes] both normalized
    overlaps = overlaps_graph(yolo_proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    # crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    # crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    # no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)                # TODO
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    # positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
    #                      config.ROI_POSITIVE_RATIO)
    # positive_count = tf.shape(positive_indices)[0]
    # positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    # positive_count = tf.shape(positive_indices)[0]

    # Negative ROIs. Add enough to maintain positive:negative ratio.
    # r = 1.0 / config.ROI_POSITIVE_RATIO
    # negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    # negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

    # Gather selected ROIs
    positive_rois = tf.gather(yolo_proposals, positive_indices)
    negative_rois = tf.gather(yolo_proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(     # assign the class number of gt_boxes (which one)
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    # roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    # deltas = mutils.box_refinement_graph(positive_rois, roi_gt_boxes)
    # deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1] (put #instance axis first)
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    x1, y1, x2, y2 = tf.split(positive_rois, 4, axis=1)
    boxes = tf.concat([y1, x1, y2, x2], axis=1)     # tf.image.crop_and_resize required this order
    # boxes = positive_rois

    # TODO: correct this
    if config.USE_MINI_MASK:
        pass
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.

        # x1, y1, x2, y2 = tf.split(positive_rois, 4, axis=1)
        # gt_x1, gt_y1, gt_x2, gt_y2 = tf.split(roi_gt_boxes, 4, axis=1)
        # gt_w = gt_x2 - gt_x1
        # gt_h = gt_y2 - gt_y1
        #
        # x1 = (x1 - gt_x1) / gt_w
        # y1 = (y1 - gt_y1) / gt_h
        # x2 = (x2 - gt_x1) / gt_w
        # y2 = (y2 - gt_y1) / gt_h
        #
        # boxes = tf.concat([y1, x1, y2, x2], axis=1)  # tf.image.crop_and_resize required

    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    # roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    # deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])  # TODO

    return rois, roi_gt_class_ids, 0, masks


class DetectMaskTargetLayer(KE.Layer):
    
    def __init__(self, config, **kwargs):
        super(DetectMaskTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        
        names = ['yolo_proposals', 'target_class_ids', 'target_bbox', 'target_mask']
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detect_mask_target_graph(
                w, x, y, z, self.config),
            self.config.BATCH_SIZE, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, input_shape[0][1], 4),  # rois
            (None, input_shape[0][1]),  # class_ids
            (None, None),               # dummy
            (None, input_shape[0][1], self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]



def build_mask_graph(rois, feature_maps, pool_size, num_classes, train_bn=False):
   
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois] + feature_maps)  # [8, ?, 14, 14, 256]

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="myolo_mask_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='myolo_mask_bn1')(x)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="myolo_mask_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name='myolo_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="myolo_mask_conv3")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name='myolo_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="myolo_mask_conv4")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name='myolo_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="myolo_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="myolo_mask")(x)
    return x


def myolo_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = K.switch(tf.size(y_true) > 0,
                    K.binary_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss

class MaskYOLO:
   

    def __init__(self,
                 mode,
                 config,
                 model_dir=None,
                 yolo_pretrain_dir=None,
                 yolo_trainable=True):
        
        assert mode in ['training', 'inference', 'yolo']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.yolo_pretrain_dir = yolo_pretrain_dir
        self.yolo_trainable = yolo_trainable
        self.keras_model = self.build(mode=mode, config=config)
        self.epoch = 0

    def build(self, mode, config):
        assert mode in ['training', 'inference', 'yolo']

        # TODO: make constraints on input image size
        w, h = config.IMAGE_SHAPE[:2]
        if w % 32 != 0 or h % 32 != 0:
            raise Exception("Image size must be dividable by 32 to adapt with YOLO framework. "
                            "For example, use 224, 256, 288, 320, 356, ... etc. ")

       
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name='input_image')

        if mode == 'training':
            # input_yolo_anchors and true_boxes
            input_true_boxes = KL.Input(shape=(1, 1, 1, config.TRUE_BOX_BUFFER, 4), name='input_true_boxes')
            input_yolo_target = KL.Input(
                shape=[config.GRID_H, config.GRID_W, config.N_BOX, 4 + 1 + config.NUM_CLASSES],
                name='input_yolo_target', dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)

            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name='input_gt_class_ids', dtype=tf.int32)

            
            input_gt_boxes = KL.Input(
                shape=[None, 4], name='input_gt_boxes', dtype=tf.float32)

            gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)

            # GT Masks (zero padded)  TODO
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[config.MINI_MASK_SHAPE[0],
                                                 config.MINI_MASK_SHAPE[1], None],
                                          name='input_gt_masks', dtype=bool)
            else:
                input_gt_masks = KL.Input(shape=[config.IMAGE_SHAPE[0],
                                                 config.IMAGE_SHAPE[1], None],
                                          name='input_gt_masks', dtype=bool)

        elif mode == 'inference':
            # true_boxes, used to compute yolo_sum_loss   TODO, useless in inference, remove it later
            input_true_boxes = KL.Input(shape=(1, 1, 1, config.TRUE_BOX_BUFFER, 4), name='input_true_boxes')

        elif mode == 'yolo':
            # true_boxes, used to compute yolo_sum_loss
            input_true_boxes = KL.Input(shape=(1, 1, 1, config.TRUE_BOX_BUFFER, 4), name='input_true_boxes')
            # y_true, used to compute yolo_sum_loss
            input_yolo_target = KL.Input(
                shape=[config.GRID_H, config.GRID_W, config.N_BOX, 4 + 1 + config.NUM_CLASSES],
                name='input_yolo_target', dtype=tf.float32)

        C4 = mobilenet_graph(input_image, config.BACKBONE, stage5=False)
        # C4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(C4)

        # decrease the depth of myolo_feature_map to make the network smaller
        myolo_feature_maps = KL.Conv2D(config.TOP_FEATURE_MAP_DEPTH, (3, 3), padding='SAME', name='feature_map')(C4)

        # build second phase of YOLO branch graph
        yolo_model = build_yolo_model(config, config.SECOND_PHASE_YOLO_DEPTH)
        yolo_output = yolo_model([C4])

        if self.yolo_pretrain_dir is not None:
            # load with pretrained yolo branch weights
            print('\nloading pretrained yolo weights --- \n')

            # YOLO Model
            inputs = [input_image, input_true_boxes, input_yolo_target]

            outputs = [yolo_output]

            model = KM.Model(inputs, outputs, name='whole_yolo_branch')
            model.load_weights(self.yolo_pretrain_dir)

            print('set the trainable of layers in the whole yolo branch as ' + str(self.yolo_trainable) + '\n')
            for layer in model.layers:
                layer.trainable = self.yolo_trainable

        # yolo_proposals = DecodeYOLOLayer(name='decode_yolo_layer', config=config)([yolo_output])

        if mode == 'training':

            yolo_proposals = DecodeYOLOLayer(name='decode_yolo_layer', config=config)([yolo_output])

            rois, target_class_ids, _, target_mask = \
                DetectMaskTargetLayer(config, name='detect_mask_targets')([
                    yolo_proposals, input_gt_class_ids, gt_boxes, input_gt_masks])

            myolo_mask = build_mask_graph(rois, [myolo_feature_maps],
                                          config.MASK_POOL_SIZE,
                                          config.NUM_CLASSES)

            # register rois as a layer in the graph
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # 1. YOLO custom loss (bbox loss and binary classification loss)
            yolo_sum_loss = KL.Lambda(lambda x: yolo_custom_loss(*x), name='yolo_sum_loss')(
                [input_yolo_target, yolo_output, input_true_boxes])

            # 2. Mask loss
            mask_loss = KL.Lambda(lambda x: myolo_mask_loss_graph(*x), name='myolo_mask_loss')(
                [target_mask, target_class_ids, myolo_mask])

            # Model
            inputs = [input_image, input_true_boxes, input_yolo_target,
                      input_gt_class_ids, input_gt_boxes, input_gt_masks]

            outputs = [yolo_output, yolo_proposals, output_rois, myolo_mask, yolo_sum_loss, mask_loss]

            model = KM.Model(inputs, outputs, name='mask+yolo')

            print('\nyolo+mask model summary: \n')
            print(model.summary())

        elif mode == "yolo":

            # YOLO custom loss (bbox loss and binary classification loss)
            yolo_sum_loss = KL.Lambda(lambda x: yolo_custom_loss(*x), name="yolo_sum_loss")(
                [input_yolo_target, yolo_output, input_true_boxes])

            # Model
            inputs = [input_image, input_true_boxes, input_yolo_target]

            outputs = [yolo_output, yolo_sum_loss]

            model = KM.Model(inputs, outputs, name="only_yolo")

            print('yolo model summary with yolo sum loss: \n')
            print(model.summary())

        elif mode == "inference":

            # Network Heads
            # Create masks for detections
            detections = DetectionsLayer(name="decode_yolo_layer", config=config)([yolo_output])
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)

            myolo_mask = build_mask_graph(detection_boxes, [myolo_feature_maps],
                                          config.MASK_POOL_SIZE,
                                          config.NUM_CLASSES)

            inputs = [input_image, input_true_boxes]
            outputs = [yolo_output, detections, myolo_mask]

            model = KM.Model(inputs, outputs, name="mask_yolo_inference")

        else:
            raise NotImplementedError

        return model

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        
        layer_regex = {
            
            "all": ".*"
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_info = []
        for id in range(0, 50):
            image, gt_class_ids, gt_boxes, gt_masks = \
                mutils.load_image_gt(train_dataset, config, id,
                                     use_mini_mask=config.USE_MINI_MASK)
            train_info.append([image, gt_class_ids, gt_boxes, gt_masks])

        val_info = []
        for id in range(0, 6):
            image, gt_class_ids, gt_boxes, gt_masks = \
                mutils.load_image_gt(val_dataset, config, id,
                                     use_mini_mask=config.USE_MINI_MASK)
            val_info.append([image, gt_class_ids, gt_boxes, gt_masks])

        train_generator = mutils.BatchGenerator(train_info, config, mode=self.mode,
                                                shuffle=True, jitter=False, norm=True)

        val_generator = mutils.BatchGenerator(val_info, config, mode=self.mode,
                                              shuffle=True, jitter=False, norm=True)


        now = datetime.datetime.now()
        tz = timezone('US/Eastern')
        fmt = '%b%d-%H-%M'
        now = tz.localize(now)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir='./', histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint('./saved_model_' + now.strftime(fmt) + '.h5', verbose=0, save_weights_only=True),
        ]

        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=len(train_generator),
            # initial_epoch=self.epoch,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            max_queue_size=3,
            verbose=1
            # workers=workers,
            # use_multiprocessing=False,
        )
        self.epoch = max(self.epoch, epochs)

    def compile(self, learning_rate, momentum):
       

        optimizer = keras.optimizers.Adam(lr=learning_rate,
                                          beta_1=0.9,
                                          beta_2=0.999,
                                          epsilon=1e-08,
                                          decay=0.0)

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}

        loss_names = ["yolo_sum_loss",  "myolo_mask_loss"]

        if self.mode == 'yolo':
            loss_names = ['yolo_sum_loss']

        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        # if verbose > 0 and keras_model is None:
        #     log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            # if trainable and verbose > 0:
            #     log("{}{:20}   ({})".format(" " * indent, layer.name,
            #                                 layer.__class__.__name__))

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

    def infer_yolo(self, image, weights_dir, save_path='./img_results/', display=True):
       
        assert list(image.shape) == config.IMAGE_SHAPE
        assert image.dtype == 'uint8'
        assert self.mode == 'yolo'

        now = datetime.datetime.now()
        tz = timezone('US/Eastern')
        fmt = '%b-%d-%H-%M'
        now = tz.localize(now)

        normed_image = image / 255.  # normalize the image to 0~1

        # form the inputs as model required
        normed_image = np.expand_dims(normed_image, axis=0)
        dummy_true_boxes = np.zeros((1, 1, 1, 1, config.TRUE_BOX_BUFFER, 4))
        dummy_target = np.zeros(shape=[1, config.GRID_H, config.GRID_W, config.N_BOX, 4 + 1 + config.NUM_CLASSES])

        # load weights
        self.load_weights(weights_dir)

        # model predict for single input image
        netout = self.keras_model.predict([normed_image, dummy_true_boxes, dummy_target])[0]

        # decode network output
        boxes = mutils.decode_one_yolo_output(netout[0],
                                              anchors=config.ANCHORS,
                                              nms_threshold=0.3,  # for shapes dataset this could be big
                                              obj_threshold=0.35,
                                              nb_class=config.NUM_CLASSES)

        normed_image = mutils.draw_boxes(normed_image[0], boxes, labels=self.config.LABELS)

        plt.imshow(normed_image[:, :, ::-1])
        plt.savefig(save_path + 'InferYOLO-' + now.strftime(fmt) + '.png')

    def detect(self, image, weights_dir, save_path='./img_results/', cs_threshold=0.35, display=True):
        
        assert list(image.shape) == config.IMAGE_SHAPE
        assert image.dtype == 'uint8'
        assert self.mode == 'inference'

        now = datetime.datetime.now()
        tz = timezone('US/Eastern')
        fmt = '%b-%d-%H-%M'
        now = tz.localize(now)

        normed_image = image / 255.  # normalize the image to 0~1

        # form the inputs as model required
        normed_image = np.expand_dims(normed_image, axis=0)
        dummy_true_boxes = np.zeros((1, 1, 1, 1, config.TRUE_BOX_BUFFER, 4))

        # load weights
        self.load_weights(weights_dir)

        # model predict for single input image
        config.BATCH_SIZE = 1
        yolo_output, detections, myolo_mask = self.keras_model.predict([normed_image, dummy_true_boxes], verbose=0)

        # test if detections align with results of yolo_output
        for detection in detections[0]:
            if detection[4] >= cs_threshold:
                print(detection)

        # decode network output
        yolo_boxes = mutils.decode_one_yolo_output(yolo_output[0],
                                                   anchors=config.ANCHORS,
                                                   nms_threshold=0.3,  # for shapes dataset this could be big
                                                   obj_threshold=0.2,
                                                   nb_class=config.NUM_CLASSES)

        results = []
        boxes, class_ids, scores, full_masks = self.decode_masks(detections, myolo_mask, image.shape)

        top10_indices = np.argsort(scores)[::-1][:10]
        list_to_remove = []
        for index in top10_indices:
            if scores[index] < cs_threshold:
                list_to_remove.append(np.where(top10_indices == index)[0][0])
        removed_indices = np.delete(top10_indices, list_to_remove)

        boxes_temp = boxes[removed_indices]
        class_ids_temp = class_ids[removed_indices]
        

        nmb_indices = mutils.NMB(boxes_temp, class_ids_temp, removed_indices, config.IMAGE_SHAPE, nms_threshold=0.7)

        nmb_indices = [109, 130]
        boxes = np.array([i * 224 for i in boxes[nmb_indices]])
        class_ids = class_ids[nmb_indices]
        scores = scores[nmb_indices]
        full_masks = full_masks[:, :, nmb_indices]

        results.append({
            "bboxes": boxes,
            "class_ids": class_ids,
            "confidence_scores": scores,
            "full_masks": full_masks,
        })

        save_path += 'InferMaskYOLO-Food-' + now.strftime(fmt) + '.png'

        if display:
            visualize.display_instances(image, boxes, full_masks, class_ids, config.LABELS, scores, save_path)

        return results

    def decode_masks(self, detections, myolo_mask, image_shape):
        

        assert len(detections) == 1    # only detect for one image per time
        assert len(myolo_mask) == 1
        assert list(image_shape) == config.IMAGE_SHAPE

       
        detection = detections[0]
        myolo_mask = myolo_mask[0]
        N = len(detection)

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detection[:N, :4]
        scores = detection[:N, 4]
        class_ids = detection[:N, 5].astype(np.int32)

        masks = myolo_mask[np.arange(N), :, :, class_ids]

         exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = mutils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks


def norm_boxes_graph(boxes, shape):
    
    w, h = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([w, h, w, h], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def trim_zeros_graph(boxes, name='trim_zeros'):
    
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


class DecodeYOLOLayer(KE.Layer):
    
    def __init__(self, config, **kwargs):
        super(DecodeYOLOLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        y_pred = inputs[0]

        cell_x = tf.to_float(
            tf.reshape(tf.tile(tf.range(config.GRID_W), [config.GRID_H]), (1, config.GRID_H, config.GRID_W, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [config.BATCH_SIZE, 1, 1, config.N_BOX, 1])

        # compute x and y based on YOLOv2 paper formula:
        # x, y = sigma(tx) + cell_grid, sigma(ty) + cell_grid
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        pred_box_xy = pred_box_xy / tf.cast(config.GRID_W, tf.float32)    # normalize to 0~1

        # compute w and h based on YOLOv2 paper formula:
        # w, h = prior_width (anchor widths) * exp(tw), prior_height (anchor heights) * exp(th)
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(config.ANCHORS, [1, 1, 1, config.N_BOX, 2])
        pred_box_wh = pred_box_wh / tf.cast(config.GRID_W, tf.float32)    # normalize

        # get diagonal coordinates, xmin, ymin, xmax, ymax
        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        # concatenate and reshape
        output_boxes = tf.concat([pred_mins, pred_maxes], axis=-1)

        output_boxes = tf.reshape(output_boxes, [-1,
                                                 output_boxes.shape[1] * output_boxes.shape[2] * output_boxes.shape[3],
                                                 output_boxes.shape[-1]])

        return output_boxes

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1] * input_shape[2] * input_shape[3], 4)


class DetectionsLayer(KE.Layer):
    
    def __init__(self, config, **kwargs):
        super(DetectionsLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        y_pred = inputs[0]
        # mask_shape = tf.shape(y_pred)[:4]
        # grid_w, grid_h = mask_shape[1], mask_shape[2]

        cell_x = tf.to_float(
            tf.reshape(tf.tile(tf.range(config.GRID_W), [config.GRID_H]), (1, config.GRID_H, config.GRID_W, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [config.BATCH_SIZE, 1, 1, config.N_BOX, 1])

        """ Adjust prediction """
        # adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        pred_box_xy = pred_box_xy / tf.cast(config.GRID_W, tf.float32)
        # adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(config.ANCHORS, [1, 1, 1, config.N_BOX, 2])
        pred_box_wh = pred_box_wh / tf.cast(config.GRID_W, tf.float32)    # normalize

        """ get x, y coordinates """
        # pred_xy = tf.expand_dims(pred_box_xy, 4)
        # pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        
         
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        pred_box_conf = tf.expand_dims(pred_box_conf, axis=-1)

        pred_box_class = tf.argmax(y_pred[..., 5:], axis=-1)
        pred_box_class = tf.expand_dims(tf.to_float(pred_box_class), axis=-1)

        detections = tf.concat([pred_mins, pred_maxes, pred_box_conf, pred_box_class], axis=-1)
        detections = tf.reshape(detections, [-1,
                                             detections.shape[1] * detections.shape[2] * detections.shape[3],
                                             detections.shape[-1]])

        return detections

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1] * input_shape[2] * input_shape[3], 4 + 1 + 1)




