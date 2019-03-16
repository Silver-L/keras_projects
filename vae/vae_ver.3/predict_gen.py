"""
* @predict generalization
* @Author: Zhihui Lu
* @Date: 2018/09/03
"""

import os
import numpy as np
import argparse
import SimpleITK as sitk
import csv
import dataIO as io
from model import Variational_auto_encoder
from keras.models import load_model, Model

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def predict_gen():
    parser = argparse.ArgumentParser(description='py, test_data_list, name_list, outdir')
    parser.add_argument('--test_data_list', '-i1', default='F:/data_info/VAE/test_data_list.txt',
                        help='test data')
    parser.add_argument('--truth_data_txt', '-i2', default='F:/data_info/VAE/test_ground_truth.txt',
                        help='name list')
    parser.add_argument('--name_list', '-i3', default='F:/data_info/VAE/test_data_name_list.txt',
                        help='name list')
    parser.add_argument('--encoder_model', '-i4', default='F:/experiment_result/vae_2/encoder_model/encoder_200.hdf5',
                        help='model')
    parser.add_argument('--decoder_model', '-i5', default='F:/experiment_result/vae_2/decoder_model/decoder_200.hdf5',
                        help='model')
    parser.add_argument('--outdir', '-i6', default='F:/experiment_result/vae_result/test', help='outdir')
    args = parser.parse_args()

    if not (os.path.exists(args.outdir)):
        os.makedirs(args.outdir)

    # load name_list
    name_list = []
    with open(args.name_list) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line: continue
            name_list.append(line[:])

    print('number of test data : {}'.format(len(name_list)))

    # load test data
    test_data = io.load_matrix_data(args.test_data_list, 'float32')
    test_data = np.expand_dims(test_data, axis=4)
    print(test_data.shape)

    # load ground truth
    ground_truth = io.load_matrix_data(args.truth_data_txt, 'int32')
    print(ground_truth.shape)

    # get image size
    image_size = []
    image_size.extend([list(test_data.shape)[1], list(test_data.shape)[2], list(test_data.shape)[3]])
    print(image_size)

    # # set network
    encoder = load_model(args.encoder_model)
    decoder = load_model(args.decoder_model)

    encoder_result = encoder.predict(test_data, 1)
    preds = decoder.predict(encoder_result, 1)

    # reshape
    preds = preds[:, :, :, :, 0]
    print(preds.shape)

    ji = []
    for i in range(preds.shape[0]):
        # # EUDT
        # eudt_image = sitk.GetImageFromArray(preds[i])
        # eudt_image.SetSpacing([1, 1, 1])
        # eudt_image.SetOrigin([0, 0, 0])
        #
        # # label
        label = np.where(preds[i] > 0, 0, 1)
        label_image = sitk.GetImageFromArray(label)
        label_image.SetSpacing([1, 1, 1])
        label_image.SetOrigin([0, 0, 0])

        # calculate ji
        ji.append([jaccard(label, ground_truth[i])])

        # # output image
        # io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(args.outdir, 'EUDT', *name_list[i])))
        # io.write_mhd_and_raw(label_image, '{}.mhd'.format(os.path.join(args.outdir, 'label', *name_list[i])))

    generalization = np.mean(ji)
    print('generalization = %f' % generalization)

    # # output csv file
    # with open(os.path.join(args.outdir, 'generalization.csv'), 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(ji)
    #     writer.writerow(['generalization= ', generalization])


def jaccard(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)

    if float(intersection.sum()) == 0.:
        return 0.
    else:
        return intersection.sum() / float(union.sum())

if __name__ == '__main__':
    predict_gen()