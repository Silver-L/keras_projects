"""
* @Variational Auto-Encoder
* @Author: Zhihui Lu
* @Date: 2018/08/28
"""

from keras.models import *
from keras.layers import *
from keras.losses import mse
import Resnet_3d as resnet


class CustomVariationalLayer(Layer):
    def set_z_mean(self, z_mean):
        self._z_mean = z_mean

    def set_z_sigma(self, z_sigma):
        self._z_sigma = z_sigma

    def _vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        reconst_loss = 368 * 304 *240 * mse(x, z_decoded)
        latent_loss = self._compute_KL_divergence(self._z_mean, self._z_sigma)
        return K.mean(reconst_loss * 0.05 + latent_loss)

    def _compute_KL_divergence(self, z_mean, z_sigma):
        return - 0.5 * K.sum(1 + K.log(K.square(z_sigma))
                                    - K.square(z_mean) - K.square(z_sigma), axis=-1)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self._vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

class Variational_auto_encoder:
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

        encodeLayer1 = Conv3D(8, 3, padding='same',
                              input_shape=(self._input_size_z, self._input_size_y, self._input_size_x, 1))(encodeLayer1)
        encodeLayer1 = BatchNormalization(axis=4)(encodeLayer1)
        encodeLayer1 = Activation('relu')(encodeLayer1)
        encodeLayer1 = AveragePooling3D(pool_size=(2, 2, 2), strides=2, padding='same')(encodeLayer1)
        encodeLayer2 = resnet._resblock(filters=8, kernel_size=3)(encodeLayer1)
        encodeLayer2 = AveragePooling3D(pool_size=(2, 2, 2), strides=2, padding='same')(encodeLayer2)
        encodeLayer3 = resnet._resblock(filters=16, kernel_size=3)(encodeLayer2)
        encodeLayer3 = AveragePooling3D(pool_size=(2, 2, 2), strides=2, padding='same')(encodeLayer3)
        encodeLayer4 = resnet._resblock(filters=32, kernel_size=3)(encodeLayer3)

        # Latent Space
        shape_before_flattening = K.int_shape(encodeLayer4)
        midLayer = Flatten()(encodeLayer4)
        midLayer = Dense(self._latent_dim, activation='relu')(midLayer)
        z_mean = Dense(self._latent_dim)(midLayer)
        z_sigma = Dense(self._latent_dim)(midLayer)

        #sampling
        Z = Lambda(self._sampling)([z_mean, z_sigma])

        # Decoder
        decoder_input = Input(K.int_shape(Z)[1:])
        decodeLayer1 = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
        decodeLayer1 = Reshape(shape_before_flattening[1:])(decodeLayer1)
        decodeLayer2 = resnet._resblock(filters=32, kernel_size=3)(decodeLayer1)
        decodeLayer2 = UpSampling3D(size=(2, 2, 2))(decodeLayer2)
        decodeLayer3 = resnet._resblock(filters=16, kernel_size=3)(decodeLayer2)
        decodeLayer3 = UpSampling3D(size=(2, 2, 2))(decodeLayer3)
        decodeLayer4 = resnet._resblock(filters=8, kernel_size=3)(decodeLayer3)
        decodeLayer4 = UpSampling3D(size=(2, 2, 2))(decodeLayer4)
        decodeLayer5 = resnet._resblock(filters=8, kernel_size=3)(decodeLayer4)

        outputs = Conv3D(1, 1)(decodeLayer5)
        outputs = UpSampling3D(size=(2, 2, 2))(outputs)
        outputs = Activation('tanh')(outputs)


        self._encoder = Model(inputs=inputs, outputs=z_mean, name='encoder')
        self._decoder = Model(inputs=decoder_input, outputs=outputs, name='decoder')

        z_decoded = self._decoder(Z)
        l = CustomVariationalLayer()
        l.set_z_mean(z_mean)
        l.set_z_sigma(z_sigma)
        y = l([inputs, z_decoded])

        self._ResVAE = Model(inputs, y, name='vae')

    def _sampling(self, args):
        z_mean, z_sigma = args
        epsilon = K.random_normal(shape=(self._latent_dim,), mean=0., stddev=1.0)
        return z_mean + z_sigma * epsilon

    def get_vae(self):
        return self._ResVAE

    def get_decoder(self):
        return self._decoder

    def get_encoder(self):
        return self._encoder
