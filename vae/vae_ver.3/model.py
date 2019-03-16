"""
* @Variational Auto-Encoder
* @Author: Zhihui Lu
* @Date: 2018/10/02
"""

from keras.models import *
from keras.layers import *
from keras.losses import mse
import tensorflow as tf

pixel_dim = 368 * 304 *240

class CustomVariationalLayer(Layer):
    def set_z_mean(self, z_mean):
        self._z_mean = z_mean

    def set_z_sigma(self, z_sigma):
        self._z_sigma = z_sigma

    def _vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        rec_loss = mse(x, z_decoded)
        latent_loss = self._compute_KL_divergence(self._z_mean, self._z_sigma)
        latent_loss = latent_loss

        return tf.reduce_mean(latent_loss * 0.0001 + rec_loss )


    def _compute_KL_divergence(self, z_mean, z_sigma):
        return - 0.5 * K.sum(1 + K.log(K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma), axis=-1)

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
        global dim
        dim = latent_dim

        # image size
        self._input_size_x = image_size[2]
        self._input_size_y = image_size[1]
        self._input_size_z = image_size[0]
        input_size = (self._input_size_z, self._input_size_y, self._input_size_x, 1)
        inputs = Input(input_size)
        print(inputs.shape)

        # Encoder
        # encodeLayer1 = Conv3D(32, kernel_size=(5, 5, 5), padding='same', strides=(2, 2, 2), input_shape=input_size)(inputs)
        # encodeLayer1 = BatchNormalization(axis=-1)(encodeLayer1)
        # encodeLayer1 = Activation('relu')(encodeLayer1)

        encodeLayer1 = AveragePooling3D(pool_size=(2, 2, 2))(inputs)

        encodeLayer2 = Conv3D(32, kernel_size=(5, 5, 5), padding='same', strides=(2, 2, 2))(encodeLayer1)
        encodeLayer2 = BatchNormalization(axis=-1)(encodeLayer2)
        encodeLayer2 = Activation('relu')(encodeLayer2)

        encodeLayer3 = Conv3D(64, kernel_size=(5, 5, 5), padding='same', strides=(2, 2, 2))(encodeLayer2)
        encodeLayer3 = BatchNormalization(axis=-1)(encodeLayer3)
        encodeLayer3 = Activation('relu')(encodeLayer3)

        encodeLayer4 = Conv3D(64, kernel_size=(5, 5, 5), padding='same', strides=(2, 2, 2))(encodeLayer3)
        encodeLayer4 = BatchNormalization(axis=-1)(encodeLayer4)
        encodeLayer4 = Activation('relu')(encodeLayer4)


        # Latent Space
        shape_before_flattening = K.int_shape(encodeLayer4)
        encodeLayer5 = Flatten()(encodeLayer4)
        z_mean = Dense(self._latent_dim)(encodeLayer5)
        z_sigma = Dense(self._latent_dim)(encodeLayer5)

        #sampling
        Z = Lambda(self._sampling)([z_mean, z_sigma])

        # Decoder
        decoder_input = Input(K.int_shape(Z)[1:])
        decodeLayer1 = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
        decodeLayer1 = Reshape(shape_before_flattening[1:])(decodeLayer1)

        decodeLayer1 = Conv3DTranspose(64, kernel_size=(5, 5, 5), padding='same', strides=(2, 2, 2))(decodeLayer1)
        decodeLayer1 = BatchNormalization(axis=-1)(decodeLayer1)
        decodeLayer1 = Activation('relu')(decodeLayer1)

        decodeLayer2 = Conv3DTranspose(32, kernel_size=(5, 5, 5), padding='same', strides=(2, 2, 2))(decodeLayer1)
        decodeLayer2 = BatchNormalization(axis=-1)(decodeLayer2)
        decodeLayer2 = Activation('relu')(decodeLayer2)

        decodeLayer3 = Conv3DTranspose(1, kernel_size=(5, 5, 5), padding='same', strides=(2, 2, 2))(decodeLayer2)
        decodeLayer3 = BatchNormalization(axis=-1)(decodeLayer3)
        decodeLayer3 = Activation('relu')(decodeLayer3)

        # decodeLayer4 = Conv3DTranspose(32, kernel_size=(5, 5, 5), padding='same', strides=(2, 2, 2))(decodeLayer3)
        # decodeLayer4 = BatchNormalization(axis=-1)(decodeLayer4)
        # decodeLayer4 = Activation('relu')(decodeLayer4)

        # outputs = Conv3DTranspose(1, kernel_size=(5, 5, 5), padding='same', activation='tanh')(decodeLayer4)

        decodeLayer4 = UpSampling3D(size=(2, 2, 2))(decodeLayer3)
        decodeLayer4 = Activation('tanh')(decodeLayer4)
        outputs = decodeLayer4

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
