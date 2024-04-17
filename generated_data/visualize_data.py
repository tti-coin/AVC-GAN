import os
import pdb
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def analize_outlier():  # TODO
    data = np.load("/workspace/outlier_val_jsai.npy")
    for i in range(len(data)):
        # 外れ値のインデックスを取得
        # outlier = tsne_obj[tsne_obj[:,0] >= 10]
        # indices = np.where(np.isin(tsne_obj, outlier))
        # # np.save("/workspace/outlier_jsai.npy",indices[0])
        # outlier_val = np.take(cat_data, indices[0], axis=0) # 外れ値のインデックスに相当するt-sne適用前の配列を取得
        # not_outlier_val = np.delete(cat_data, indices[0], axis=0)
        # # np.save("/workspace/outlier_val_jsai.npy",outlier_val)
        # np.save("/workspace/not_outlier_val_jsai.npy",not_outlier_val)
        # pdb.set_trace()
        fig = plt.figure()
        plt.plot(data[i], linewidth=2)
        plt.savefig(
            f"/workspace/generated_data/anal_outlier_jsai/outlier_jsai_{i+1}.png"
        )
        plt.clf()
        plt.close(fig)
