"""
* @Trainer
* @Author: Zhihui Lu
* @Date: 2018/07/21
"""

import os
import numpy as np
import argparse
import dataIO as io
from random import randint
import matplotlib.pyplot as plt
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.optimizers import Adam
from keras import losses
from keras.callbacks import ModelCheckpoint

from model import Autoencoder


def main():
    parser = argparse.ArgumentParser(description='py, train_data_txt, train_data_ture_txt, validation_data_txt, outdir')
    parser.add_argument('--train_data_txt', '-i1', default='F:/data_info/autoencoder/train_data_list.txt',
                        help='train data list')

    parser.add_argument('--train_ground_truth_txt', '-i2',
                        default='F:/data_info/autoencoder/train_ground_truth.txt',
                        help='train ground truth list')

    parser.add_argument('--validation_data_txt', '-i3', default='F:/data_info/autoencoder/validation_data_list.txt',
                        help='validation data list')

    parser.add_argument('--validation_ground_truth_txt', '-i4',
                        default='F:/data_info/autoencoder/validation_ground_truth.txt',
                        help='validation ground truth list')

    parser.add_argument('--outdir', '-i5', default='D:/M1_research/autoencoder/result/test', help='outdir')
    args = parser.parse_args()

    # check folder
    if not (os.path.exists(args.outdir)):
        os.mkdir(args.outdir)

    # define
    batch_size = 3
    epoch = 2500

    # load train data
    train_data = io.load_matrix_data(args.train_data_txt, 'float32')
    train_data = np.expand_dims(train_data, axis=4)

    # load train ground truth
    train_truth = io.load_matrix_data(args.train_ground_truth_txt, 'float32')
    train_truth = np.expand_dims(train_truth, axis=4)

    # load validation data
    val_data = io.load_matrix_data(args.validation_data_txt, 'float32')
    val_data = np.expand_dims(val_data, axis=4)

    # load validation ground truth
    val_truth = io.load_matrix_data(args.validation_ground_truth_txt, 'float32')
    val_truth = np.expand_dims(val_truth, axis=4)

    print(' number of training: {}'.format(len(train_data)))
    print('size of traning: {}'.format(train_data.shape))
    print(' number of validation: {}'.format(len(val_data)))
    print('size of validation: {}'.format(val_data.shape))

    image_size = []
    image_size.extend([list(train_data.shape)[1], list(train_data.shape)[2], list(train_data.shape)[3]])

    # set network
    network = Autoencoder(*image_size)
    model = network.model()
    model.summary()
    model.compile(optimizer='Nadam', loss=losses.mean_squared_error, metrics=['mse'])

    # set data_set
    train_steps, train_data = batch_iter(train_data, train_truth, batch_size)
    valid_steps, val_data = batch_iter(val_data, val_truth, batch_size)

    # fit network
    model_checkpoint = ModelCheckpoint(os.path.join(args.outdir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                       verbose=1)

    history = model.fit_generator(train_data, steps_per_epoch=train_steps, epochs=epoch, validation_data=val_data,
                                  validation_steps=valid_steps, verbose=1, callbacks=[model_checkpoint])

    plot_history(history, args.outdir)


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
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    fig = plt.legend(['loss', 'val_loss'], loc='lower right')
    filename = open(os.path.join(path,'loss.pickle'), 'wb')
    pickle.dump(fig, filename)
    plt.show()

if __name__ == '__main__':
    main()
