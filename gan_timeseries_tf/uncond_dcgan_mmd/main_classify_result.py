"""
Copyright 2018 The Matrix Authors
This file is part of the Matrix library.

The Matrix library is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Matrix library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the Matrix library. If not, see <http://www.gnu.org/licenses/>.
@author: Steve Deng
"""

import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':

    in_dir = "/home/hfl/dataset/output/gan_timeseries_tf/uncond_dcgan_base_mmd/"
    out_dir = "/home/hfl/dataset/output/gan_timeseries_tf/uncond_dcgan_base_mmd/000-result"

    with open("{}/classification_accuracy.md".format(in_dir), 'r') as f:
        lines = f.readlines()
        names = []
        knn, knn_gan, lsvc, lsvc_gan, lr, lr_gan = [], [], [], [], [], []
        for line in lines[2::]:
            values = line.strip().split('|')
            names.append(values[0])
            knn.append(float(values[1].strip()))
            knn_gan.append(float(values[2].strip()))
            lsvc.append(float(values[3].strip()))
            lsvc_gan.append(float(values[4].strip()))
            lr.append(float(values[5].strip()))
            lr_gan.append(float(values[6].strip()))

        names = np.array(names)
        knn = np.array(knn)
        knn_gan = np.array(knn_gan)
        lsvc = np.array(lsvc)
        lsvc_gan = np.array(lsvc_gan)
        lr = np.array(lr)
        lr_gan = np.array(lr_gan)

        # accuracy compare
        fig = plt.figure()
        plt.scatter(knn, knn_gan)
        plt.plot(np.array([0, 1]), np.array([0, 1]))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('1NN')
        plt.ylabel('1NN-TSGAN')
        plt.savefig(os.path.join(out_dir, "0000_acc_knn.png"))

        fig = plt.figure()
        plt.scatter(lsvc, lsvc_gan)
        plt.plot(np.array([0, 1]), np.array([0, 1]))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('LinearSVM')
        plt.ylabel('LinearSVM-TSGAN')
        plt.savefig(os.path.join(out_dir, "0000_acc_lsvc.png"))
        plt.close(fig)

        fig = plt.figure()
        plt.scatter(lr, lr_gan)
        plt.plot(np.array([0, 1]), np.array([0, 1]))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('LR')
        plt.ylabel('LR-TSGAN')
        plt.savefig(os.path.join(out_dir, "0000_acc_lr.png"))
        plt.close(fig)

        # count the number of acc_gan larger than acc
        names_knn_large = []
        names_knn_less = []
        names_lsvc_large = []
        names_lsvc_less = []
        names_lr_large = []
        names_lr_less = []
        for i in np.arange(len(knn)):
            if knn_gan[i] >= knn[i]:
                names_knn_large.append(names[i])
            else:
                names_knn_less.append(names[i])
            if lsvc_gan[i] >= lsvc[i]:
                names_lsvc_large.append(names[i])
            else:
                names_lsvc_less.append(names[i])
            if lr_gan[i] >= lr[i]:
                names_lr_large.append(names[i])
            else:
                names_lr_less.append(names[i])
        with open(os.path.join(out_dir, "0000_acc"), 'w') as f:
            knn_large_str = ''.join(str(e) for e in names_knn_large)
            knn_less_str = ''.join(str(e) for e in names_knn_less)
            lsvc_large_str = ''.join(str(e) for e in names_lsvc_large)
            lsvc_less_str = ''.join(str(e) for e in names_lsvc_less)
            lr_large_str = ''.join(str(e) for e in names_lr_large)
            lr_less_str = ''.join(str(e) for e in names_lr_less)

            f.write("knn_gan_larger: " + knn_large_str + "\n")
            f.write("knn_gan_lesser: " + knn_less_str + "\n")
            f.write("lsvc_gan_larger: " + lsvc_large_str + "\n")
            f.write("lsvc_gan_lesser: " + lsvc_less_str + "\n")
            f.write("lr_gan_larger: " + lr_large_str + "\n")
            f.write("lr_gan_lesser: " + lr_less_str + "\n")
