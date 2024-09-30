import os
import pdb
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.manifold import TSNE
from torch import optim

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class Exp_VAE(Exp_Basic):
    def __init__(self, args):
        super(Exp_VAE, self).__init__(args)

    def _build_model(self):
        model = (
            self.model_dict[self.args.ae_model].Model(self.args, self.device).float()
        )
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def loss_function(self, outputs, batch_x, mu, logvar):
        recon_loss = F.mse_loss(outputs, batch_x, reduction="mean")
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss, kl_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # CHANGED
                dec_inp = (
                    torch.zeros(
                        self.args.ae_batch_size,
                        self.args.seq_len + self.args.label_len,
                        self.args.enc_in,
                    )
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs, mu, logvar = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]

                mu_cpu = mu.detach().cpu()
                logvar_cpu = logvar.detach().cpu()
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                recon_loss, cl_loss = self.loss_function(pred, true, mu_cpu, logvar_cpu)
                loss = 0.9 * recon_loss + 0.1 * cl_loss  # TODO
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        _, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        # criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # make preprocessed data
        prepro_dir = (
            f"./data/preprocessed_datasets/{self.args.des}/sl{self.args.seq_len}"
        )
        if not os.path.exists(os.path.join(prepro_dir)):
            os.makedirs(prepro_dir)

            prepro_data = []
            for j, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                prepro_data.append(batch_x.cpu().numpy())
            prepro_data = np.vstack(prepro_data)
            np.save(os.path.join(prepro_dir, "prepro_train_shuffled.npy"), prepro_data)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            recon_losses = []
            kl_losses = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = (
                    torch.zeros(
                        self.args.ae_batch_size,
                        (self.args.seq_len + self.args.label_len),
                        self.args.enc_in,
                    )
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark) #, mu, logvar
                            outputs, mu, logvar = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        loss = self.loss_function(
                            outputs, batch_x, mu, logvar
                        )  # CHANGED
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]
                    else:
                        outputs, mu, logvar = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    recon_loss, kl_loss = self.loss_function(
                        outputs, batch_x, mu, logvar
                    )  # CHANGED

                    loss = 0.9 * recon_loss + 0.1 * kl_loss  # TODO
                    train_loss.append(loss.item())
                    recon_losses.append(recon_loss.item())
                    kl_losses.append(kl_loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # train_loss = np.average(train_loss)
            train_recon_loss = np.average(recon_losses)
            train_kl_loss = np.average(kl_losses)
            vali_loss = self.vali(vali_data, vali_loader, None)
            test_loss = self.vali(test_data, test_loader, None)

            print(
                "Epoch: {0}, Steps: {1} | Train Recon Loss: {2:.7f} Train KL Loss: {3:.7f} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                    epoch + 1,
                    train_steps,
                    train_recon_loss,
                    train_kl_loss,
                    vali_loss,
                    test_loss,
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if "PEMS" in self.args.data or "Solar" in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = (
                    torch.zeros(
                        self.args.ae_batch_size,
                        self.args.seq_len + self.args.label_len,
                        self.args.enc_in,
                    )
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )[0]

                    else:
                        outputs, _, _ = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                outputs = outputs.detach().cpu().numpy()
                batch_x = batch_x.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(
                        shape
                    )
                    batch_x = test_data.inverse_transform(batch_x.squeeze(0)).reshape(
                        shape
                    )

                pred = outputs
                true = batch_x

                preds.append(pred)
                trues.append(true)

        preds = np.array(preds)
        trues = np.array(trues)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))
        f = open("result_long_term_forecast.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n")
        f.write("\n")
        f.close()

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)

        return

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag="pred")

    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path + "/" + "checkpoint.pth"
    #         self.model.load_state_dict(torch.load(best_model_path))

    #     preds = []

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
    #             pred_loader
    #         ):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
    #             dec_inp = (
    #                 torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
    #                 .float()
    #                 .to(self.device)
    #             )
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if self.args.output_attention:
    #                         outputs = self.model(
    #                             batch_x, batch_x_mark, dec_inp, batch_y_mark
    #                         )[0]
    #                     else:
    #                         outputs = self.model(
    #                             batch_x, batch_x_mark, dec_inp, batch_y_mark
    #                         )
    #             else:
    #                 if self.args.output_attention:
    #                     outputs = self.model(
    #                         batch_x, batch_x_mark, dec_inp, batch_y_mark
    #                     )[0]
    #                 else:
    #                     outputs = self.model(
    #                         batch_x, batch_x_mark, dec_inp, batch_y_mark
    #                     )
    #             outputs = outputs.detach().cpu().numpy()
    #             if pred_data.scale and self.args.inverse:
    #                 shape = outputs.shape
    #                 outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(
    #                     shape
    #                 )
    #             preds.append(outputs)

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    #     # result save
    #     folder_path = "./results/" + setting + "/"
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     np.save(folder_path + "real_prediction.npy", preds)

    #     return

    def save_recon_as_npy(self, setting, load=False):
        """
        generate trues and recons for evaluation
        """
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        train_vali_data = [train_data, vali_data]
        train_vali_loader = [train_loader, vali_loader]

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))
            print("Loaded trained AE model to reconstruct data.")

        self.model.eval()
        with torch.no_grad():
            for flag, loader in enumerate(train_vali_loader):
                real_hiddens = []
                recons = []
                trues = []
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                    loader
                ):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = (
                        torch.zeros(
                            self.args.ae_batch_size,
                            (self.args.seq_len + self.args.label_len),
                            self.args.enc_in,
                        )
                        .float()
                        .to(self.device)
                    )
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                enc_out = self.model.encode(batch_x)
                                outputs = self.model.decode(enc_out)[0]
                                # outputs = self.model(
                                #     batch_x, batch_x_mark, dec_inp, batch_y_mark
                                # )[0]
                            else:
                                enc_out = self.model.encode(batch_x)
                                outputs = self.model.decode(enc_out)
                    else:
                        if self.args.output_attention:
                            enc_out = self.model.encode(batch_x)
                            outputs = self.model.decode(enc_out)[0]

                        else:
                            enc_out, _, _ = self.model.encode(batch_x)
                            outputs = self.model.decode(enc_out)

                    enc_out = enc_out.detach().cpu().numpy()
                    outputs = outputs.detach().cpu().numpy()
                    batch_x = batch_x.detach().cpu().numpy()
                    if (train_data.scale or vali_data.scale) and self.args.inverse:
                        shape = outputs.shape

                        outputs = (
                            train_vali_data[flag]
                            .inverse_transform(outputs.squeeze(0))
                            .reshape(shape)
                        )
                        batch_x = (
                            train_vali_data[flag]
                            .inverse_transform(batch_x.squeeze(0))
                            .reshape(shape)
                        )
                    real_hidden = enc_out
                    recon = outputs
                    true = batch_x

                    real_hiddens.append(real_hidden)
                    recons.append(recon)
                    trues.append(true)

                real_hiddens = np.array(real_hiddens)
                recons = np.array(recons)
                trues = np.array(trues)

                real_hiddens = real_hiddens.reshape(
                    -1, real_hiddens.shape[-2], real_hiddens.shape[-1]
                )
                recons = recons.reshape(-1, recons.shape[-2], recons.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

                # result save
                folder_path = os.path.join(self.args.checkpoints, setting, "eval_ae/")
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                flag_name = "train" if flag == 0 else "vali"
                np.save(folder_path + f"real_hiddens_{flag_name}.npy", real_hiddens)
                np.save(folder_path + f"recons_{flag_name}.npy", recons)
                np.save(folder_path + f"trues_{flag_name}.npy", trues)

    # plot recnstracted data with tsne and matplotlib
    def plot_recon_as_tsne(self, setting):
        save_dir = os.path.join("./checkpoints/", setting, "eval_ae/")

        flags = ["train", "vali"]
        for flag in flags:
            ori_data = np.load(os.path.join(save_dir, f"trues_{flag}.npy"))
            recon_data = np.load(os.path.join(save_dir, f"recons_{flag}.npy"))

            # codebase: TimeGAN
            anal_sample_no = min([1000, len(ori_data), len(recon_data)])
            np.random.seed(0)
            ori_idx = np.random.permutation(anal_sample_no)[:anal_sample_no]

            ori_data = ori_data[ori_idx]
            recon_data = recon_data[ori_idx]

            for i in range(min(recon_data.shape[2], 10)):
                cat_data = np.concatenate(
                    [ori_data[:, :, i], recon_data[:, :, i]], axis=0
                )

                print(f"Plotting reconstructed {flag} data with matplotlib")
                outputs_dir = os.path.join("checkpoints", setting, "eval_ae/outputs")
                if not os.path.exists(outputs_dir):
                    os.makedirs(outputs_dir)

                for i in range(min(recon_data.shape[2], 10)):
                    fig = plt.figure(figsize=(20, 20))
                    for j in range(9):
                        fig.add_subplot(3, 3, (j + 1))
                        plt.plot(ori_data[j, :, i], label="trues", linewidth=1)
                        plt.plot(recon_data[j, :, i], label="recons", linewidth=1)
                    plt.legend()
                    plt.title(f"reconstructed {flag} data")
                    plt.savefig(
                        os.path.join(outputs_dir, f"list_data_{flag}_ch{i}.png"),
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    # if not self.args.no_wandb:
                    #     wandb.log({f"eval/reconstruted/ch{i}/{flag}": wandb.Image(plt)})
                    plt.clf()
                    plt.close(fig)

    def plot_multi_hidden_as_tsne(self, ae_setting):
        data_dir = os.path.join("./checkpoints/", ae_setting, "eval_ae/")
        save_dir = os.path.join("./checkpoints/", ae_setting, "eval_ae/")
        real_hiddens = np.load(os.path.join(data_dir, "real_hiddens_train.npy"))

        anal_sample_no = min([1000, len(real_hiddens)])
        np.random.seed(0)
        idx = np.random.permutation(anal_sample_no)[:anal_sample_no]
        real_hiddens = real_hiddens[idx]

        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        f, ax = plt.subplots(1)

        multi_cat_data = real_hiddens.reshape([-1, real_hiddens.shape[2]])
        tsne = TSNE(
            n_components=2, random_state=0, verbose=1, perplexity=40, n_iter=300
        )
        tsne_obj = tsne.fit_transform(multi_cat_data)

        for i in range(real_hiddens.shape[1]):
            plt.scatter(
                tsne_obj[i * anal_sample_no : (i + 1) * anal_sample_no, 0],
                tsne_obj[i * anal_sample_no : (i + 1) * anal_sample_no, 1],
                alpha=0.2,
                label=f"ch{i}",
                color=cycle[i],
                s=5,
            )

            ax.legend()
        plt.title(f"t-SNE plot of train multi hidden states")
        plt.xlabel("x-tsne")
        plt.ylabel("y-tsne")
        plt.savefig(
            os.path.join(save_dir, f"tsne_multi_hidden_train.png"),
            bbox_inches="tight",
            pad_inches=0,
        )
