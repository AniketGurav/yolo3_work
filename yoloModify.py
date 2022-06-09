import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU, ZeroPadding2D, BatchNormalization, MaxPool2D
from tensorflow.keras.regularizers import l2
from yolov3.configs import *
from yolov3.yolov3 import *
from yolov3.utils import read_class_names
import tensorflow.keras.backend as K

def darknet53(input_data):
    input_data = convolutional(input_data, (3, 3,  3,  32))
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)

    input_data = convolutional(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)

    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data

def decode(conv_output, NUM_CLASS, i=0):
    # where i = 0, 1 or 2 to correspond to the three grid scales
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    #conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS+604+165))# phosc change

    if TRAIN_PHOSC:
        conv_output = tf.reshape(conv_output,(batch_size, output_size, output_size, 3, 5 + NUM_CLASS + 604 + 165))  # phosc change
    elif TRAIN_PHOC:
        conv_output = tf.reshape(conv_output,(batch_size, output_size, output_size, 3, 5 + NUM_CLASS + 604))  # phosc change
    elif TRAIN_PHOS:
        conv_output = tf.reshape(conv_output,(batch_size, output_size, output_size, 3, 5 + NUM_CLASS + 165))  # phosc change
    else:
        conv_output = tf.reshape(conv_output,(batch_size, output_size, output_size, 3, 5 + NUM_CLASS ))  # phosc change

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
    conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
    conv_raw_prob = conv_output[:, :, :, :, 5:5 + NUM_CLASS ] # category probability of the prediction box # phosc change

    if TRAIN_PHOSC:
        conv_raw_phoc = conv_output[:, :, :, :, 5 + NUM_CLASS:5 + NUM_CLASS+604] #  phosc change
        conv_raw_phos = conv_output[:, :, :, :, 5 + NUM_CLASS+604:] #  phosc change
    elif TRAIN_PHOC:
        conv_raw_phoc = conv_output[:, :, :, :, 5 + NUM_CLASS:5 + NUM_CLASS+604] #  phosc change
    elif TRAIN_PHOS:
        conv_raw_phos = conv_output[:, :, :, :, 5 + NUM_CLASS+604:] #  phosc change

    # next need Draw the grid. Where output_size is equal to 13, 26 or 52
    y = tf.range(output_size, dtype=tf.int32)
    y = tf.expand_dims(y, -1)
    y = tf.tile(y, [1, output_size])
    x = tf.range(output_size,dtype=tf.int32)
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [output_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    # Calculate the center position of the prediction box:
    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
    # Calculate the length and width of the prediction box:
    pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = tf.sigmoid(conv_raw_conf) # object box calculates the predicted confidence
    pred_prob = tf.sigmoid(conv_raw_prob) # calculating the predicted probability category box object

    if TRAIN_PHOSC:
        pred_phoc_prob = tf.sigmoid(conv_raw_phoc)  # calculating the predicted probability phoc
        pred_phos_prob = tf.keras.activations.relu(conv_raw_phos)

    elif TRAIN_PHOC:
        pred_phoc_prob = tf.sigmoid(conv_raw_phoc)  # calculating the predicted probability phoc
        pred_phos_prob = []
    elif TRAIN_PHOS:
        pred_phoc_prob = []
        pred_phos_prob = tf.keras.activations.relu(conv_raw_phos)

    #pred_phos_prob = tf.relu(conv_raw_phoc)  # calculating the predicted probability category box object

    if TRAIN_PHOSC or TRAIN_PHOS:
        pred_phos_prob =tf.keras.activations.relu(conv_raw_phos)

    else:
        pred_phos_prob=[]
    # calculating the predicted probability category box object


    if TRAIN_PHOSC:
        return tf.concat([pred_xywh, pred_conf, pred_prob,pred_phoc_prob,pred_phos_prob], axis=-1)
    elif TRAIN_PHOC:
        return tf.concat([pred_xywh, pred_conf, pred_prob,pred_phoc_prob], axis=-1)
    elif TRAIN_PHOS:
        return tf.concat([pred_xywh, pred_conf, pred_prob,pred_phos_prob], axis=-1)

def getShape(indx,nm,tensor):

    print(" indx:",indx, "name:",nm," shape:",tensor.shape)


def yolo33(input_size=416, CLASSES=YOLO_COCO_CLASSES,channels=3, training=False):
    CLASSES = TRAIN_CLASSES
    NUM_CLASS = len(read_class_names(CLASSES))
    #print("**")
    input_layer = Input([416, 416, 3])

    route_1, route_2, conv = darknet53(input_layer)
    print("route_1:",route_1.shape)
    print("route_2:",route_2.shape)
    print("conv:",conv.shape)
    #def   convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):

    conv = convolutional(conv, (1, 1, 1024, 512), name="dark_cov1")
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv = convolutional(conv, (3, 3, 512, 1024))
    conv = convolutional(conv, (1, 1, 1024, 512))
    conv_lobj_branch = convolutional(conv, (3, 3, 512, 1024), name="dark_conv_lobj")

    getShape(1,"conv_lobj_branch",conv_lobj_branch)

    if TRAIN_PHOSC:
        conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5 + 769)), activate=False, bn=False,
                                   name="dark_conv_lobj2")  # phosc change
    # l2: (None, 13, 13, 2325)

    elif TRAIN_PHOC:
        conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5 + 604)), activate=False, bn=False,
                                   name="dark_conv_lobj2")  # phosc change

    elif TRAIN_PHOS:
        conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5 + 165)), activate=False, bn=False,
                                   name="dark_conv_lobj2")  # phosc change
    else:
        # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255]
        conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3*(NUM_CLASS + 5)), activate=False, bn=False)


    conv = convolutional(conv, (1, 1, 512, 256))
    # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
    # upsampling process does not need to learn, thereby reducing the network parameter
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)
    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv_mobj_branch = convolutional(conv, (3, 3, 256, 512), name="dark_conv_mobj")

    # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]


    if TRAIN_PHOSC:
        conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5 + 769)), activate=False, bn=False,
                                   name="dark_conv_mobj2")  # phosc change

    elif TRAIN_PHOC:
        conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5 + 604)), activate=False, bn=False,
                                   name="dark_conv_mobj2")  # phosc change

    elif TRAIN_PHOS:
        conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5 + 165)), activate=False, bn=False,
                                   name="dark_conv_mobj2")  # phosc change
    else:
        conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5 )), activate=False, bn=False,
                                   name="dark_conv_mobj2")


    #print("\n\t conv_mbbox=",conv_mbbox.shape)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)

    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv_sobj_branch = convolutional(conv, (3, 3, 128, 256), name="dark_conv_sobj")

    if TRAIN_PHOSC:
       conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5 + 769)), activate=False, bn=False,
                                  name="dark_conv_sobj2")  ## phosc change
    elif TRAIN_PHOC:
       conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5 + 604)), activate=False, bn=False,
                                  name="dark_conv_sobj2")  ## phosc change

    elif TRAIN_PHOS:
       conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5 + 165)), activate=False, bn=False,
                                  name="dark_conv_sobj2")  ## phosc change
    else:
       conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5 )), activate=False, bn=False,
                                  name="dark_conv_sobj2")


    conv_tensors = [conv_sbbox, conv_mbbox, conv_lbbox]

    output_tensors = []

    for ii, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, NUM_CLASS, ii)
        if training:
            output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    Yolo = tf.keras.Model(input_layer, output_tensors)
    return Yolo

if __name__ == '__main__':
     print("888")
     img = np.random.random((1,416, 416, 3))
     yolo=yolo33()
     pred_result =yolo(img)
     print("--",pred_result[0].shape)
     print("--",pred_result[1].shape)
     print("--",pred_result[2].shape)
