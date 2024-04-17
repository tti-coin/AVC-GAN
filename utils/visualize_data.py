import os
import pdb
import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_recon_data(setting):
    recons = np.load(os.path.join("checkpoints", setting, "eval_ae/recons.npy"))
    trues = np.load(os.path.join("checkpoints", setting, "eval_ae/trues.npy"))
    save_dir = os.path.join("checkpoints", setting, "eval_ae/outputs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    label_len = 48  # TODO: ラベルの扱い方
    if recons.shape[2] != 1:
        for i in range(recons.shape[2]):
            for j in range(10):
                fig = plt.figure()
                plt.plot(recons[j,:,i], label="recons", linewidth=2)
                plt.plot(trues[j,label_len:,i], label="trues", linewidth=2)
                plt.legend()
                plt.savefig(os.path.join(save_dir, f"sample_variate{i}_no{j}.png"))
                plt.clf()
                plt.close(fig)
    else:
        for i in range(10):
            fig = plt.figure()
            plt.plot(recons[i], label="recons", linewidth=2)
            plt.plot(trues[i][label_len:], label="trues", linewidth=2)
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"sample_{i}.png"))
            plt.clf()
            plt.close(fig)


def plot_recon_tsne(ae_setting):
    label_len = 48  # TODO: ラベルの扱い方
    save_dir = os.path.join(
        "/workspace/checkpoints/", ae_setting, "eval_ae/"
    )
    ori_data = np.load(os.path.join(save_dir, "trues.npy"))
    recon_data = np.load(os.path.join(save_dir, "recons.npy"))
    ori_data = ori_data[:1000]
    recon_data = recon_data[:1000]

    cat_data = np.concatenate([ori_data[:,label_len:,-1], recon_data[:,:,-1]], axis=0)

    tsne = TSNE(n_components=2, random_state=0, verbose=1, perplexity=40, n_iter=300)
    tsne_obj = tsne.fit_transform(cat_data)

    f, ax = plt.subplots(1)
    plt.scatter(
        tsne_obj[: len(ori_data), 0],
        tsne_obj[: len(ori_data), 1],
        alpha=0.2,
        label="Original",
    )
    plt.scatter(
        tsne_obj[len(ori_data) :, 0],
        tsne_obj[len(ori_data) :, 1],
        alpha=0.2,
        label="Reconstructed",
    )

    ax.legend()
    plt.title("t-SNE plot of reconstructed data")
    plt.xlabel("x-tsne")
    plt.ylabel("y-tsne")
    plt.savefig(os.path.join(save_dir, "tsne_recon.png"))
    # plt.savefig(f'/workspace/tsne-jsai-label.pdf')
    # plt.show()

def plot_hidden_tsne(ae_setting, gan_setting):
    save_dir = os.path.join(
        "/workspace/checkpoints/", ae_setting, gan_setting, "eval_gan/"
    )
    ori_data = np.load(os.path.join(save_dir, "real_reps.npy"))
    gen_data = np.load(os.path.join(save_dir, "hiddens.npy"))

    ori_data = np.squeeze(ori_data)
    ori_data = ori_data[: len(gen_data)]
    ori_data = ori_data[:1000]
    gen_data = gen_data[:1000]
    print("number of samples: ", len(ori_data))

    cat_data = np.concatenate([ori_data, gen_data], axis=0)

    tsne = TSNE(n_components=2, random_state=0, verbose=1, perplexity=40, n_iter=300)
    tsne_obj = tsne.fit_transform(cat_data)

    f, ax = plt.subplots(1)
    plt.scatter(
        tsne_obj[: len(ori_data), 0],
        tsne_obj[: len(ori_data), 1],
        alpha=0.2,
        label="Original",
    )
    plt.scatter(
        tsne_obj[len(ori_data) :, 0],
        tsne_obj[len(ori_data) :, 1],
        alpha=0.2,
        label="Generated",
    )

    ax.legend()
    plt.title("t-SNE plot of hidden states")
    plt.xlabel("x-tsne")
    plt.ylabel("y-tsne")
    plt.savefig(os.path.join(save_dir, "tsne_hidden.png"))
    # plt.savefig(f'/workspace/tsne-jsai-label.pdf')
    # plt.show()


def plot_dec_tsne(ae_setting, gan_setting):
    save_dir = os.path.join(
        "/workspace/checkpoints/", ae_setting, gan_setting, "generated_data/"
    )
    ori_data = pickle.load(open("/workspace/data/preprocessed_for_eval/ori_ettm2-288-jsai.pkl", "rb"))
    ori_data = ori_data.squeeze()
    gen_dataset = h5py.File(os.path.join(save_dir, "data.h5"), "r")
    gen_data = gen_dataset["chunk_00000"]

    ori_data = np.squeeze(ori_data)
    ori_data = ori_data[: len(gen_data)]
    ori_data = ori_data[:1000]
    gen_data = gen_data[:1000]
    print("number of samples: ", len(ori_data))

    cat_data = np.concatenate([ori_data, gen_data], axis=0)

    tsne = TSNE(n_components=2, random_state=0, verbose=1, perplexity=40, n_iter=300)
    tsne_obj = tsne.fit_transform(cat_data)

    f, ax = plt.subplots(1)
    plt.scatter(
        tsne_obj[: len(ori_data), 0],
        tsne_obj[: len(ori_data), 1],
        alpha=0.2,
        label="Original",
    )
    plt.scatter(
        tsne_obj[len(ori_data) :, 0],
        tsne_obj[len(ori_data) :, 1],
        alpha=0.2,
        label="Generated",
    )

    ax.legend()
    plt.title("t-SNE plot of decoded data")
    plt.xlabel("x-tsne")
    plt.ylabel("y-tsne")
    plt.savefig(os.path.join(save_dir, "tsne_dec.png"))


def plot_gen_data(ae_setting, gan_setting):
    save_dir = os.path.join("/workspace/checkpoints/", ae_setting, gan_setting, "generated_data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gen_dataset = h5py.File(os.path.join(save_dir, "data.h5"), "r")
    gen_data = gen_dataset["chunk_00000"]

    for i in range(10):
        fig = plt.figure()
        plt.plot(gen_data[i], linewidth=2)  # color='#03af7a'
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.savefig(os.path.join(save_dir, f"synth_{i}.png"))
        # plt.savefig(f'/workspace/generated_data/anal_outlier_jsai/not_outlier_jsai_{i+1}.png')
        # plt.savefig(f'/workspace/generated_data/anal_outlier_jsai/not_outlier_jsai_{i+1}.pdf')
        plt.clf()
        plt.close(fig)
