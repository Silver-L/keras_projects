"""
* @Auto-encoder
* @Author: Zhihui Lu
* @Date: 2018/07/21
"""

from keras.models import *
from keras.layers import *
import tensorflow as tf


class Autoencoder:
    def __init__(self, size_z, size_y, size_x):
        self.input_size_x = size_x
        self.input_size_y = size_y
        self.input_size_z = size_z

        input_size = (self.input_size_z, self.input_size_y, self.input_size_x, 1)
        inputs = Input(input_size)
        print(inputs.shape)

        # Encoder
        encodeLayer1 = self.__add_encode_layers(8, 3, inputs, is_first=True)
        encodeLayer2 = MaxPooling3D(pool_size=(2, 2, 2))(encodeLayer1)
        encodeLayer3 = self.__add_encode_layers(16, 3, encodeLayer2)
        encodeLayer4 = MaxPooling3D(pool_size=(2, 2, 2))(encodeLayer3)

        midLayer = self.__add_encode_layers(16, 3, encodeLayer4)

        # Decoder
        decodeLayer1 = self.__add_decode_layers(16, 3, midLayer, upsamping=True)
        decodeLayer2 = self.__add_decode_layers(16, 3, decodeLayer1)
        decodeLayer3 = self.__add_decode_layers(8, 3, decodeLayer2, upsamping=True)
        decodeLayer4 = Conv3D(1, 1)(decodeLayer3)

        # upsamping to origin size

        outputs = UpSampling3D(size=(2, 2, 2))(decodeLayer4)

        print(outputs)

        self.MODEL = Model(inputs=[inputs], outputs=[outputs])

    def __add_encode_layers(self, filter_size, kernel_size, input_layer, is_first=False):
        layer = input_layer
        if is_first:
            layer = Conv3D(filter_size, kernel_size, padding='same',
                           input_shape=(self.input_size_z, self.input_size_y, self.input_size_x, 1))(layer)
        else:
            layer = Conv3D(filter_size, kernel_size, padding='same')(layer)

        layer = BatchNormalization()(layer)
        layer = Activation(activation='relu')(layer)
        print(layer.shape)
        return layer

    def __add_decode_layers(self, filter_size, kernel_size, input_layer, upsamping=False):
        layer = input_layer
        if upsamping:
            layer = UpSampling3D(size=(2, 2, 2))(layer)

        layer = Conv3D(filter_size, kernel_size, padding='same')(layer)
        layer = BatchNormalization()(layer)
        layer = Activation(activation='relu')(layer)
        print(layer.shape)
        return layer

    def model(self):
        return self.MODEL
