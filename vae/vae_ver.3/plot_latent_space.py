"""
* @plot latent space
* @Author: Zhihui Lu
* @Date: 2018/09/04
"""

import os
import numpy as np
import argparse
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import pickle
import dataIO as io

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def plot_latent_space():
    parser = argparse.ArgumentParser(description='py, test_data_list, name_list, outdir')
    parser.add_argument('--model', '-i1', default='',
                        help='model')
    parser.add_argument('--train_data_list', '-i2', default='',
                        help='name list')
    parser.add_argument('--outdir', '-i3', default='', help='outdir')
    args = parser.parse_args()

    if not (os.path.exists(args.outdir)):
        os.makedirs(args.outdir)


    # load test data
    train_data = io.load_matrix_data(args.train_data_list, 'float32')
    train_data = np.expand_dims(train_data, axis=-1)
    print(train_data.shape)
    image_size = []
    image_size.extend([list(train_data.shape)[1], list(train_data.shape)[2], list(train_data.shape)[3]])

    # set network

    encoder = load_model(args.model)
    latent_space = encoder.predict(train_data, batch_size=1)

    plt.figure(figsize=(8, 6))
    fig = plt.scatter(latent_space[:, 0], latent_space[:, 1])
    plt.title('latent distribution')
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    filename = open(os.path.join(args.outdir, 'latent_distribution.pickle'), 'wb')
    pickle.dump(fig, filename)
    plt.show()


if __name__ == '__main__':
    plot_latent_space()
