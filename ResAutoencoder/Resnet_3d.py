"""
* @Resblock 3d
* @Author: Zhihui Lu
* @Date: 2018/08/22
"""

from keras.layers import *
from keras.regularizers import l2

global DIM1_AXIS
global DIM2_AXIS
global DIM3_AXIS
global channel_axis

DIM1_AXIS = 1
DIM2_AXIS = 2
DIM3_AXIS = 3
channel_axis = 4

def _bn_relu(input):
    norm = BatchNormalization(axis=channel_axis)(input)
    return Activation("relu")(norm)

def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1,1,1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f

def _conv_bn(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1,1,1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(input)
        return BatchNormalization(axis=channel_axis)(conv)

    return f

def _shortcut3d(input, residual):
    stride_dim1 = input._keras_shape[DIM1_AXIS] \
        // residual._keras_shape[DIM1_AXIS]
    stride_dim2 = input._keras_shape[DIM2_AXIS] \
        // residual._keras_shape[DIM2_AXIS]
    stride_dim3 = input._keras_shape[DIM3_AXIS] \
        // residual._keras_shape[DIM3_AXIS]
    equal_channels = residual._keras_shape[channel_axis] \
        == input._keras_shape[channel_axis]

    shortcut = input
    if stride_dim1 != 1 or stride_dim2 != 1 or stride_dim3 !=1 \
        or not equal_channels:
        shortcut = Conv3D(
            filters=residual._keras_shape[channel_axis],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal", padding="valid",
            kernel_regularizer=l2(1e-4)
        )(input)
    return add([shortcut, residual])


def _resblock(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1,1,1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1e-4))

    def f(input):
        output = _conv_bn_relu(filters=filters, kernel_size=kernel_size,
                               strides=strides, kernel_initializer=kernel_initializer,
                               padding=padding, kernel_regularizer=kernel_regularizer)(input)

        output = _conv_bn(filters=filters, kernel_size=kernel_size,
                          strides=strides, kernel_initializer=kernel_initializer,
                          padding=padding, kernel_regularizer=kernel_regularizer)(output)

        output = _shortcut3d(input,output)
        return Activation("relu")(output)

    return f