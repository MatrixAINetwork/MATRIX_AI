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
from scipy.misc import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_series(samples, out_path):
    f, axes = plt.subplots(samples.shape[0] // 2, 2)
    axes = axes.flat[:]
    #title = out_path.split('/')[-1].split('.')[0]
    #plt.suptitle(title)
    for i, ax in enumerate(axes):
        ax.plot(samples[i])

    plt.savefig(out_path)
    plt.close(f)

def plot_gan_loss(g_losses, d_losses, save_path):
    fig = plt.figure()
    plt.title('loss')
    plt.plot(g_losses, label='g_loss')
    plt.plot(d_losses, label='d_loss')
    plt.xlabel('index')
    plt.ylabel('loss')
    plt.legend(loc='best')
    plt.savefig(save_path)
    plt.close(fig)


def plot_dic(dic_values, title=None, xlabel=None, ylabel=None,ylim=None, save_path=None):
    fig = plt.figure()
    for key, y in dic_values.items():
        plt.plot(y, label=key)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if ylim:
        plt.ylim(ylim)
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)
