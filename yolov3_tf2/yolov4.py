from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from .utils import broadcast_iou

flags.DEFINE_integer('yolo_max_boxes', 100,
                     'maximum number of boxes per image')
flags.DEFINE_float('yolo_iou_threshold', 0.3, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.6, 'score threshold')

# For YoloV3 ######################
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
####################################
# For YoloV4 ######################
yolov4_anchors = np.array([(12, 16), (19, 36), (40,28), (36,75), (76,55),(72,146), (142,110), (192,243), (459,401)], np.float32) / 608
yolov4_anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
xyscale = np.array([1.2, 1.1, 1.05])
strides = np.array([8, 16, 32])
####################################
# MOT v3
#yolo_anchors = np.array([(0.03154305, 0.03636364), (0.02962515, 0.08181818),
#                         (0.02552083, 0.12037037), (0.03495311, 0.10681818),
#                         (0.03239557, 0.13148148), (0.04262575, 0.12685185),
#                         (0.03748489, 0.14629632), (0.04688832, 0.16203704),
#                         (0.06770833, 0.28240741)],
#                        np.float32)

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])

def Yolo(nclass, model='yolov4', is_tiny=False):
    if is_tiny:
        if model == 'yolov4':
            return YoloV4_tiny(classes=nclass)
        elif model == 'yolov3':
            return YoloV3_tiny(classes=nclass)
    else:
        if model == 'yolov4':
            return YoloV4(classes=nclass)
        elif model == 'yolov3':
            return YoloV3(classes=nclass)

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

def DarknetConv(x, filters, size, strides=1, batch_norm=True, activation=True, activation_type='Leaky'):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
    if activation:
        if activation_type == "Leaky":
            x = LeakyReLU(alpha=0.1)(x)
        elif activation_type == "mish":
            x = mish(x)
    return x

def cspDarknetResidual(x, filters, block_1):
    prev = x
    x = DarknetConv(x, filters, 1, activation_type='mish')
    if block_1:
        x = DarknetConv(x, filters * 2, 3, activation_type='mish')
    else:
        x = DarknetConv(x, filters, 3, activation_type='mish')
    x = Add()([prev, x])
    return x

def cspDarknetBlock(x, filters, blocks, block_1=False):
    if block_1:
        filters_2 = filters
    else:
        filters_2 = filters // 2
    x = DarknetConv(x, filters, 3, strides=2, activation_type='mish')
    x_route = DarknetConv(x, filters_2, 1, activation_type='mish')
    x = DarknetConv(x, filters_2, 1, activation_type='mish')
    for _ in range(blocks):
        x = cspDarknetResidual(x, filters // 2, block_1)
    x = DarknetConv(x, filters_2, 1, activation_type='mish')
    x = Concatenate()([x, x_route])
    x = DarknetConv(x, filters, 1, activation_type='mish')
    return x

def cspdarknet53(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3, activation_type='mish')  #1
    x = cspDarknetBlock(x, 64, 1, block_1=True)     #11
    x = cspDarknetBlock(x, 128, 2)                  #24
    x = x54 = cspDarknetBlock(x, 256, 8)                #55 -> route 54
    x = x85 = cspDarknetBlock(x, 512, 8)                #86 -> route 85
    x = cspDarknetBlock(x, 1024, 4)                 #105
    return tf.keras.Model(inputs, (x54, x85, x), name=name)

def SPP(filters, name=None):
    def spp_enhance(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x_6 = DarknetConv(x, filters, 1)
        ## SPP ##
        x_5 = MaxPool2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x_6)
        x_3 = MaxPool2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x_6)
        x_1 = MaxPool2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x_6)
        x = Concatenate()([x_1, x_3, x_5, x_6])
        return Model(inputs, x, name=name)(x_in)
    return spp_enhance

def PAN(filters, name=None):
    def pan_neck(x_in):
        inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:]), Input(x_in[2].shape[1:])
        x, x85, x54 = inputs
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = x_37 = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters // 2, 1)
        x = UpSampling2D(2)(x)
        x85 = DarknetConv(x85, filters // 2, 1)
        # concat with skip connection (x_85)
        x = Concatenate()([x85, x])
        x = DarknetConv(x, filters // 2, 1)
        x = DarknetConv(x, filters, 3)
        x = DarknetConv(x, filters // 2, 1)

        x = DarknetConv(x, filters, 3)
        x = x_16 = DarknetConv(x, filters // 2, 1)
        x = DarknetConv(x, filters // 4, 1)
        x = UpSampling2D(2)(x)
        x54 = DarknetConv(x54, filters // 4, 1)
        # concat with skip connection (x_54)
        x = Concatenate()([x54, x])
        return Model(inputs, (x_37, x_16, x), name=name)(x_in)
    return pan_neck


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x

def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x

def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)

def DarknetTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)

def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv

def YoloV4Conv(filters, name=None):
    def yolov4_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs
            # concat with skip connection
            x = DarknetConv(x, filters, 3, strides=2)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolov4_conv

def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv

def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output

def decode(box_xywh, scores, score_threshold=0.4, input_shape = tf.constant([416,416])):
    batch_size = tf.shape(conv_output)[0]
    conv_output = tf.reshape(conv_output,
                             (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = ((tf.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = tf.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = tf.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob

def yolov4_boxes(pred, anchors, classes, xyscale, strides):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    #grid = tf.tile(tf.expand_dims(grid, axis=0), [tf.shape(pred)[0], 1, 1, 3, 1])

    box_xy = ((box_xy * xyscale) - 0.5 * (xyscale - 1) + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32) # * strides
    #
    box_wh = tf.exp(box_wh) * anchors

    #bbox = tf.concat([box_xy, box_wh], axis=-1)
    box_x1y1 = box_xy - box_wh / 2
    box_y1x1 = box_x1y1[..., ::-1]
    box_x2y2 = box_xy + box_wh / 2
    box_y2x2 = box_x2y2[..., ::-1]
    bbox = tf.concat([box_y1x1, box_y2x2], axis=-1)

    return bbox, objectness, class_probs, pred_box

def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_y1x1 = box_x1y1[..., ::-1]
    box_x2y2 = box_xy + box_wh / 2
    box_y2x2 = box_x2y2[..., ::-1]
    bbox = tf.concat([box_y1x1, box_y2x2], axis=-1)

    return bbox, objectness, class_probs, pred_box

def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections


def filter_boxes(outputs, anchors, masks, classes, score_threshold=0.25, input_shape=tf.constant([608,608], tf.float32)):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
    # concatenate detections from scales
    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)
    #
    scores = confidence * class_probs
    #
    scores_max = tf.math.reduce_max(scores, axis=-1)
    #
    mask = scores_max >= score_threshold
    #
    class_boxes = tf.boolean_mask(bbox, mask)
    pred_conf = tf.boolean_mask(scores, mask)
    class_boxes = tf.reshape(class_boxes, [tf.shape(bbox)[0], -1, tf.shape(class_boxes)[-1]])
    pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])
    #
    #box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)
    #box_yx = box_xy[..., ::-1]
    #box_hw = box_wh[..., ::-1]

    #box_mins = (box_yx - (box_hw / 2.)) / input_shape
    #box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    #boxes = tf.concat([
    #    box_mins[..., 0:1],  # y_min
    #    box_mins[..., 1:2],  # x_min
    #    box_maxes[..., 0:1],  # y_max
    #    box_maxes[..., 1:2]  # x_max
    #], axis=-1)
    #
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(class_boxes, (tf.shape(class_boxes)[0], -1, 1, 4)),
        scores=pred_conf,
        max_output_size_per_class=FLAGS.yolo_max_boxes,
        max_total_size=FLAGS.yolo_max_boxes,
        iou_threshold=FLAGS.yolo_iou_threshold,
        score_threshold=FLAGS.yolo_score_threshold
    )
    return boxes, scores, classes, valid_detections


def YoloV4(size=None, channels=3, anchors=yolov4_anchors,
           masks=yolov4_anchor_masks, xyscale=xyscale, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')
    # Backbone
    x54, x85, x = cspdarknet53(name='yolo_cspdarknet53')(x)
    # Spatial Enhancer
    x = SPP(512, name='yolo_spp')(x)
    # Neck
    x_37, x_16, x = PAN(512, name='yolo_pan')((x, x85, x54))
    # Head
    x = YoloV4Conv(128, name='yolo_conv_0')(x)
    output_0 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_0')(x)

    x = YoloV4Conv(256, name='yolo_conv_1')((x, x_16))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloV4Conv(512, name='yolo_conv_2')((x, x_37))
    output_2 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov4')

    boxes_0 = Lambda(lambda x: yolov4_boxes(x, anchors[masks[0]],  classes, xyscale[0], strides[0]),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolov4_boxes(x, anchors[masks[1]], classes, xyscale[1], strides[1]),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolov4_boxes(x, anchors[masks[2]], classes, xyscale[2], strides[2]),
                     name='yolo_boxes_2')(output_2)
    #
    outputs = Lambda(lambda x: filter_boxes(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))
    return Model(inputs, outputs, name='yolov4')

def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')

def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='yolo_darknet')(x)

    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')

def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)

        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)

        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss
