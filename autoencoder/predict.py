"""
* @predict
* @Author: Zhihui Lu
* @Date: 2018/07/24
"""

import os
import numpy as np
import argparse
import dataIO as io
from model import Autoencoder
import SimpleITK as sitk

# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def predict():
    parser = argparse.ArgumentParser(description='py, test_data_list, name_list, outdir')
    parser.add_argument('--test_data_list', '-i1', default='F:/data_info/autoencoder/test_data_list.txt',
                        help='test data')
    parser.add_argument('--name_list', '-i2', default='F:/data_info/autoencoder/test_data_name_list.txt',
                        help='name list')
    parser.add_argument('--model', '-i3', default='D:/M1_research/autoencoder/result/model/weights.2372-0.43.hdf5',
                        help='model')
    parser.add_argument('--outdir', '-i4', default='D:/M1_research/autoencoder/result/predict', help='outdir')
    args = parser.parse_args()

    if not (os.path.exists(args.outdir)):
        os.mkdir(args.outdir)

    # load name_list
    name_list = []
    with open(args.name_list) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line: continue
            name_list.append(line[:])

    print('number of test data : {}'.format(len(name_list)))

    test_data = io.load_matrix_data(args.test_data_list, 'float32')
    test_data = np.expand_dims(test_data, axis=4)
    print(test_data.shape)

    image_size = []
    image_size.extend([list(test_data.shape)[1], list(test_data.shape)[2], list(test_data.shape)[3]])
    print(image_size)

    # set network
    network = Autoencoder(*image_size)
    model = network.model()
    model.load_weights(args.model)

    preds = model.predict(test_data, 1)
    preds = preds[:, :, :, :, 0]

    print(preds.shape)

    for i in range(preds.shape[0]):
        # EUDT
        eudt_image = sitk.GetImageFromArray(preds[i])
        eudt_image.SetSpacing([1, 1, 1])
        eudt_image.SetOrigin([0, 0, 0])

        # label
        label = np.where(preds[i] > 0, 0, 1)
        label_image = sitk.GetImageFromArray(label)
        label_image.SetSpacing([1, 1, 1])
        label_image.SetOrigin([0, 0, 0])

        io.write_mhd_and_raw(eudt_image, '{}.mhd'.format(os.path.join(args.outdir, 'EUDT', *name_list[i])))
        io.write_mhd_and_raw(label_image, '{}.mhd'.format(os.path.join(args.outdir, 'label', *name_list[i])))

if __name__ == '__main__':
    predict()