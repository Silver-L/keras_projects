"""
* @predict specificity
* @Author: Zhihui Lu
* @Date: 2018/09/03
"""

import os
import time
import numpy as np
import argparse
import SimpleITK as sitk
import csv
from keras import backend as K
from keras.models import load_model
import dataIO as io

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def predict_spe():
    parser = argparse.ArgumentParser(description='py, test_data_list, name_list, outdir')
    parser.add_argument('--model', '-i1', default='',
                        help='model')
    parser.add_argument('--truth_data_txt', '-i2', default='',
                        help='name list')
    parser.add_argument('--outdir', '-i3', default='', help='outdir')
    args = parser.parse_args()

    if not (os.path.exists(args.outdir)):
        os.mkdir(args.outdir)

    latent_dim = 32
    n_gen = 1

    # load ground truth
    ground_truth = io.load_matrix_data(args.truth_data_txt, 'int32')
    print(ground_truth.shape)

    # aset network
    decoder = load_model(args.model)

    specificity = []
    for i in range(n_gen):

        # # generate shape
        # sample_z = np.full(latent_dim, 0)
        # sample_z = np.array([sample_z])
        sample_z = np.random.normal(0, 1.0, (1, latent_dim))
        # print(sample_z.shape)
        preds = decoder.predict(sample_z)
        preds = preds[:, :, :, :, 0]

        # # EUDT
        eudt_image = sitk.GetImageFromArray(preds[0])
        eudt_image.SetSpacing([1, 1, 1])
        eudt_image.SetOrigin([0, 0, 0])

        # label
        label = np.where(preds[0] > 0, 0, 1)
        label_image = sitk.GetImageFromArray(label)
        label_image.SetSpacing([1, 1, 1])
        label_image.SetOrigin([0, 0, 0])

        # calculate ji
        case_max_ji = 0.
        for image_index in range(ground_truth.shape[0]):
            ji = jaccard(label, ground_truth[image_index])
            if ji > case_max_ji:
                case_max_ji = ji
        specificity.append([case_max_ji])

        # output image
        io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(args.outdir, 'EUDT', str(i+1))))
        io.write_mhd_and_raw(label_image, '{}.mhd'.format(os.path.join(args.outdir, 'label', str(i+1))))

    # output csv file
    # with open(os.path.join(args.outdir, 'specificity.csv'), 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(specificity)
    #     writer.writerow(['specificity:', np.mean(specificity)])

    print('specificity = %f' % np.mean(specificity))


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


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
    start = time.time()
    predict_spe()
    process_time = time.time() - start

    print(process_time)
