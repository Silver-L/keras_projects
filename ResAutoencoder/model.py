"""
* @ResBlock Variational Auto-Encoder
* @Author: Zhihui Lu
* @Date: 2018/08/22
"""

from keras.models import *
from keras.layers import *
from keras.losses import mse
import Resnet_3d as resnet


class Res_Variational_auto_encoder:
    def __init__(self, latent_dim, *image_size):
        # latent dim
        self._latent_dim = latent_dim

        # image size
        self._input_size_x = image_size[2]
        self._input_size_y = image_size[1]
        self._input_size_z = image_size[0]

        input_size = (self._input_size_z, self._input_size_y, self._input_size_x, 1)
        inputs = Input(input_size)
        print(inputs.shape)

        # Encoder
        encodeLayer1 = AveragePooling3D(pool_size=(2, 2, 2), strides=2, padding='same')(inputs)
        encodeLayer1 = Conv3D(8, 3, padding='same')(encodeLayer1)
        encodeLayer1 = AveragePooling3D(pool_size=(2, 2, 2), strides=2, padding='same')(encodeLayer1)
        encodeLayer1 = BatchNormalization(axis=4)(encodeLayer1)
        encodeLayer2 = resnet._resblock(filters=8, kernel_size=3)(encodeLayer1)
        encodeLayer2 = AveragePooling3D(pool_size=(2, 2, 2), strides=2, padding='same')(encodeLayer2)
        encodeLayer3 = resnet._resblock(filters=16, kernel_size=3)(encodeLayer2)
        encodeLayer3 = AveragePooling3D(pool_size=(2, 2, 2), strides=2, padding='same')(encodeLayer3)
        encodeLayer4 = resnet._resblock(filters=32, kernel_size=3)(encodeLayer3)

        # Latent Space
        shape_before_flattening = K.int_shape(encodeLayer4)
        midLayer = Flatten()(encodeLayer4)
        midLayer = Dense(self._latent_dim, activation='relu')(midLayer)
        decodeLayer1 = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(midLayer)
        decodeLayer1 = Reshape(shape_before_flattening[1:])(decodeLayer1)

        decodeLayer2 = resnet._resblock(filters=16, kernel_size=3)(decodeLayer1)
        decodeLayer2 = UpSampling3D(size=(2, 2, 2))(decodeLayer2)
        decodeLayer3 = resnet._resblock(filters=16, kernel_size=3)(decodeLayer2)
        decodeLayer3 = UpSampling3D(size=(2, 2, 2))(decodeLayer3)
        decodeLayer4 = resnet._resblock(filters=8, kernel_size=3)(decodeLayer3)
        decodeLayer4 = UpSampling3D(size=(2, 2, 2))(decodeLayer4)
        decodeLayer5 = resnet._resblock(filters=8, kernel_size=3)(decodeLayer4)
        decodeLayer6 = Conv3D(1, 1)(decodeLayer5)

        # #upsamping to origin size
        outputs = UpSampling3D(size=(2, 2, 2))(decodeLayer6)
        outputs = Activation('tanh')(outputs)
        self.MODEL = Model(inputs=[inputs], outputs=[outputs])

    def model(self):
        return self.MODEL
