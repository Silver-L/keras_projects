"""
* @Trainer
* @Author: Zhihui Lu
* @Date: 2018/08/30
"""

import os
import argparse
import dataIO as io
import matplotlib.pyplot as plt
import pickle
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.optimizers import Adam
from keras.models import *

from model import Variational_auto_encoder


def main():
    parser = argparse.ArgumentParser(description='py, train_data_txt, train_data_ture_txt, validation_data_txt, outdir')

    parser.add_argument('--train_data_txt', '-i1', default='F:/data_info/VAE/train_data_list.txt',
                        help='train data list')

    parser.add_argument('--validation_data_txt', '-i2', default='F:/data_info/VAE/validation_data_list.txt',
                        help='validation data list')

    parser.add_argument('--outdir', '-i3', default='F:/experiment_result/vae_2', help='outdir')

    args = parser.parse_args()

    # check folder
    if not (os.path.exists(args.outdir + '/encoder_model')):
        os.makedirs(args.outdir + '/encoder_model')
    if not (os.path.exists(args.outdir + '/decoder_model')):
        os.makedirs(args.outdir + '/decoder_model')

    # define
    batch_size = 3
    epoch = 200
    latent_dim = 2

    # load train data
    train_data = io.load_matrix_data(args.train_data_txt, 'float32')
    train_data = np.expand_dims(train_data, axis=4)

    # load validation data
    val_data = io.load_matrix_data(args.validation_data_txt, 'float32')
    val_data = np.expand_dims(val_data, axis=4)


    print(' number of training: {}'.format(len(train_data)))
    print('size of traning: {}'.format(train_data.shape))
    print(' number of validation: {}'.format(len(val_data)))
    print('size of validation: {}'.format(val_data.shape))

    image_size = []
    image_size.extend([list(train_data.shape)[1], list(train_data.shape)[2], list(train_data.shape)[3]])

    # # set network
    network = Variational_auto_encoder(latent_dim, *image_size)
    model = network.get_vae()
    encoder = network.get_encoder()
    decoder = network.get_decoder()
    model.summary()
    model.compile(optimizer=Adam(lr=8e-4, beta_1=0.5, beta_2=0.9), loss=[zero_loss])

    # set data_set
    train_steps, train_data = batch_iter(train_data, train_data, batch_size)
    valid_steps, val_data = batch_iter(val_data, val_data, batch_size)

    # fit network
    history_total = []
    for epoch_index in range(epoch):
        history = model.fit_generator(train_data, steps_per_epoch=train_steps, epochs=1, validation_data=val_data,
                                  validation_steps=valid_steps, verbose=1)

        history_total.append(history)
        encoder.save(os.path.join(args.outdir + '/encoder_model/', 'encoder_{}.hdf5'.format(epoch_index + 1)))
        decoder.save(os.path.join(args.outdir + '/decoder_model/', 'decoder_{}.hdf5'.format(epoch_index + 1)))

    plot_history(history_total, args.outdir)

def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

# data generator
def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X = shuffled_data[start_index: end_index]
                y = shuffled_labels[start_index: end_index]
                yield X, y

    return num_batches_per_epoch, data_generator()

# plot loss
def plot_history(history, path):
    train_loss = []
    val_loss = []
    epoch = []
    for index in range(len(history)):
        train_loss.append(history[index].history['loss'])
        val_loss.append(history[index].history['val_loss'])
        epoch.append(index + 1)

    plt.xticks(np.arange(1, len(history) + 5, int((len(history) + 5) / 10)))
    plt.plot(epoch, train_loss)
    plt.plot(epoch, val_loss)
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    fig = plt.legend(['loss', 'val_loss'], loc='lower right')
    filename = open(os.path.join(path, 'loss.pickle'), 'wb')
    pickle.dump(fig, filename)
    plt.show()


if __name__ == '__main__':
    main()
