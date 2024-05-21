"""
Gradient Penalty 部分の実装を変更
"""

import os
import pdb
import time
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.manifold import TSNE
from torch import autograd, optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from model.gan import Discriminator, Generator
from model.iTransformer import Model
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class Exp_iTransGAN(Exp_Basic):
    def __init__(self, args):
        super(Exp_iTransGAN, self).__init__(args)
        self.device = self._acquire_device()
        self.generator = Generator(
            self.args.enc_in, self.args.noise_dim, self.device
        ).to(
            self.device
        )  # FIXME
        print(self.generator)
        if self.args.use_hidden:
            self.discriminator = Discriminator(
                self.args.enc_in, self.args.noise_dim, self.device
            ).to(self.device)
        # self.discriminator = Discriminator(self.args.noise_dim).to(self.device)
        else:
            self.discriminator = Discriminator(
                self.args.enc_in, self.args.pred_len, self.device
            ).to(self.device)
        print(self.discriminator)

        self.discriminator_optm = torch.optim.RMSprop(
            params=self.discriminator.parameters(),
            lr=self.args.gen_lr,
            alpha=self.args.gan_alpha,
        )
        self.generator_optm = torch.optim.RMSprop(
            params=self.generator.parameters(),
            lr=self.args.disc_lr,
            alpha=self.args.gan_alpha,
        )

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def load_ae(self, setting):
        self.ae.load_state_dict(
            torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
        )

    def load_generator(self, pretrained_dir=None):
        if pretrained_dir is not None:
            path = pretrained_dir
        else:
            path = "{}/generator.dat".format(self.params["root_dir"])
        self.logger.info("load: " + path)
        self.generator.load_state_dict(torch.load(path, map_location=self.device))

    def train_gan(self, ae_setting, gan_setting):
        print("Loading trained AutoEncoder model")
        self.model.load_state_dict(
            torch.load(
                os.path.join("/workspace/checkpoints/", ae_setting, "checkpoint.pth")
            )
        )

        _, train_loader = self._get_data(flag="train")
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        self.discriminator.train()
        self.generator.train()
        self.model.train()  # TODO

        save_dir = os.path.join("/workspace/checkpoints/", ae_setting, gan_setting)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        gan_iter = self.args.gan_iter
        d_update = self.args.d_update

        for iteration in range(gan_iter):
            avg_d_loss = 0
            t1 = time.time()

            toggle_grad(self.generator, False)
            toggle_grad(self.discriminator, True)
            self.generator.train()
            self.discriminator.train()

            # train discriminator
            for j in range(d_update):
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                    train_loader
                ):
                    _, seq_len, N = batch_x.size()
                    self.discriminator_optm.zero_grad()
                    z = torch.randn(
                        self.args.gan_batch_size, self.args.enc_in, self.args.noise_dim
                    ).to(self.device)

                    real_rep = self.model.encode(batch_x.float().to(self.device))
                    if self.args.use_hidden:
                        d_real = self.discriminator(real_rep)
                    else:
                        real_dec = self.model.decode(
                            real_rep
                        )  # real_dec: (batch_size, seq_len, N)
                        d_real = self.discriminator(torch.permute(real_dec, (0, 2, 1)))
                    # d_real = self.discriminator(real_rep)

                    # On fake data
                    with torch.no_grad():
                        x_fake = self.generator(z)

                    x_fake.requires_grad_()
                    # x_fake = torch.unsqueeze(x_fake, dim=1)
                    # fake_dec = self.model.decode(x_fake)
                    # fake_dec = torch.squeeze(fake_dec)
                    if self.args.use_hidden:
                        d_fake = self.discriminator(x_fake)
                    else:
                        fake_dec = self.model.decode(x_fake)
                        d_fake = self.discriminator(torch.permute(fake_dec, (0, 2, 1)))

                    # get gradient penalty
                    if self.args.use_hidden:
                        gradient_penalty = self.grad_penalty(real_rep, x_fake)
                    else:
                        gradient_penalty = self.grad_penalty(real_dec, fake_dec)

                    d_loss = d_fake.mean() - d_real.mean() + gradient_penalty
                    d_loss.backward()

                    self.discriminator_optm.step()
                    avg_d_loss += (d_fake.mean() - d_real.mean()).item()
                    break

            avg_d_loss /= d_update

            # train generator
            toggle_grad(self.generator, True)
            toggle_grad(self.discriminator, False)
            self.generator.train()
            self.discriminator.train()
            self.generator_optm.zero_grad()
            z = torch.randn(
                self.args.gan_batch_size, self.args.enc_in, self.args.noise_dim
            ).to(self.device)
            fake = self.generator(z)

            if self.args.use_hidden:
                g_loss = -self.discriminator(fake).mean()
            else:
                fake_data = self.model.decode(fake)
                g_loss = -self.discriminator(torch.permute(fake_data, (0, 2, 1))).mean()
            g_loss.backward()
            self.generator_optm.step()

            if (iteration + 1) % 10 == 0:
                print(
                    "[Iteration: %d/%d] [Time: %f] [D_loss: %f] [G_loss: %f] [gp: %f]"
                    % (
                        iteration + 1,
                        gan_iter,
                        time.time() - t1,
                        avg_d_loss,
                        g_loss.item(),
                        gradient_penalty.item(),
                    )
                )

            if not self.args.no_wandb:
                log_dict = {
                    "train/loss/d_loss": avg_d_loss,
                    "train/loss/d_loss_real": d_real.mean(),
                    "train/loss/d_loss_fake": d_fake.mean(),
                    "train/loss/g_loss": g_loss.item(),
                    "train/loss/gradient_penalty": gradient_penalty.item(),
                }
                wandb.log(log_dict)

            if (iteration + 1) % 2000 == 0:
                dec_out = self.model.decode(fake)
                gen_data = (
                    dec_out[:, -self.args.pred_len :, :]
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )

                for i in range(min(gen_data.shape[2], 10)):
                    fig = plt.figure(figsize=(20, 20))
                    for j in range(16):
                        fig.add_subplot(4, 4, (j + 1))
                        plt.plot(gen_data[j, :, i], label="generated", linewidth=1)
                    plt.legend()

                    if not self.args.no_wandb:
                        print("logging generated data")
                        wandb.log(
                            {f"train/generated_per_step/ch{i}": wandb.Image(plt)},
                            (iteration + 1),
                        )
                    plt.clf()
                    plt.close(fig)

            if (iteration + 1) % 5000 == 0:
                print(f"Save {iteration+1}iter WGAN model")
                torch.save(
                    self.generator.state_dict(),
                    os.path.join(
                        save_dir,
                        f"generator_iter{iteration+1}.dat",
                    ),
                )

            # if (iteration + 1) % self.dev_step == 0:
            #     if dev_dataset is None:
            #         self.logger.warning("development dataset is not specified")
            #     else:
            #         # compute dev score
            #         self.logger.info("compute dev score")
            #         dev_batch = DataSetIter(dataset=dev_dataset, batch_size=self.args.gan_batch_size, sampler=RandomSampler())
            #         dev_score = 0.0
            #         dev_real_score = 0.0
            #         dev_fake_score = 0.0
            #         with torch.no_grad():
            #             for batch_x, batch_y in dev_batch:
            #                 # loss of real
            #                 sta = None
            #                 dyn = batch_x["dyn"].to(self.device)
            #                 seq_len = batch_x["seq_len"].to(self.device)
            #                 real_rep = self.ae.encoder(sta, dyn, seq_len)
            #                 d_real = self.discriminator(real_rep) # input representation to discriminator
            #                 dloss_real = -d_real.mean()
            #                 # loss of fake
            #                 z = torch.randn(self.args.gan_batch_size, self.params["noise_dim"]).to(self.device)
            #                 fake = self.generator(z)
            #                 d_fake = self.discriminator(fake)
            #                 dloss_fake = d_fake.mean()
            #                 # compute dev score
            #                 dev_score += dloss_real.item() + dloss_fake.item()
            #                 dev_real_score += dloss_real.item()
            #                 dev_fake_score += dloss_fake.item()
            #         log_dict["gan_dev/d_loss"] = dev_score
            #         log_dict["gan_dev/d_loss_real"] = dev_real_score
            #         log_dict["gan_dev/d_loss_fake"] = dev_fake_score

            # if self.wandb is not None:
            #     self.wandb.log(log_dict)

        print(f"Save {iteration + 1}iter WGAN model")
        torch.save(
            self.generator.state_dict(),
            os.path.join(
                save_dir,
                f"generator_iter{iteration + 1}.dat",
            ),
        )

    def save_hiddens_and_generated_as_npy(self, ae_setting, gan_setting):
        """
        Save hidden states

        Parameters:
        ----------
        ae_setting : str
        gan_setting : str

        Returns:
        ----------
        None : None
        """
        self.model.load_state_dict(
            torch.load(os.path.join("./checkpoints/", ae_setting, "checkpoint.pth"))
        )
        # self.generator.load_state_dict(
        #     torch.load(
        #         os.path.join(
        #             "./checkpoints/",
        #             ae_setting,
        #             gan_setting,
        #             f"generator_iter{self.args.load_iter}.dat",
        #         )
        #     )
        # )
        self.generator.load_state_dict(
            torch.load(
                "/home/user/workspace/checkpoints/variemb_multi_lr0001_bsz32_iTransformer_ETTm2_ftM_sl432_ll48_pl288_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_ettm2_projection_0/debug_cgan_multi_hidden_gp_gbsz1024_glr0.0001_dlr0.0001_nd128_du1_hdTrue/generator_iter3.dat"
            )
        )
        print("DEBUG MODE")

        print(f"Loading trained {self.args.load_iter} WGAN model")
        self.model.eval()
        self.generator.eval()

        # generate hidden states and save them
        with torch.no_grad():
            z = torch.randn(
                self.args.gan_batch_size, self.args.enc_in, self.args.noise_dim
            ).to(self.device)
            x_fake = self.generator(z)
            hiddens = x_fake.detach().cpu().numpy()
            dec_fake = self.model.decode(x_fake)
            generated = dec_fake[:, -self.args.pred_len :, :].squeeze().cpu().numpy()

        save_dir = os.path.join("./checkpoints/", ae_setting, gan_setting, "eval_gan/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(
            os.path.join(save_dir, "hiddens.npy"), hiddens
        )  ### FIXME: shape (batch, noise_dim, enc_in)

        save_data_dir = os.path.join(
            "./checkpoints/",
            ae_setting,
            gan_setting,
            f"generated_data_iter{self.args.load_iter}",
        )
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        np.save(
            os.path.join(save_data_dir, "sample_data.npy"), generated
        )  # shape (batch, pred_len, N)

    def save_real_reps_as_npy(self, ae_setting, gan_setting):

        save_dir = os.path.join("./checkpoints/", ae_setting, gan_setting, "eval_gan/")
        _, train_loader = self._get_data(flag="train")

        real_reps = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):

                enc_out = self.model.encode(batch_x.float().to(self.device))
                real_rep = enc_out.detach().cpu().numpy()
                real_reps.append(real_rep)

        real_reps = np.array(real_reps)
        real_reps = real_reps.reshape(-1, real_reps.shape[-2], real_reps.shape[-1])

        np.save(
            os.path.join(save_dir, "real_reps.npy"), real_reps
        )  # shape (batch*len(train_loader), enc_in, noise_dim)

    def plot_hidden_tsne(self, ae_setting, gan_setting):
        save_dir = os.path.join("./checkpoints/", ae_setting, gan_setting, "eval_gan/")
        ori_data = np.load(os.path.join(save_dir, "real_reps.npy"))
        gen_data = np.load(os.path.join(save_dir, "hiddens.npy"))

        # ori_data = np.squeeze(ori_data)
        # ori_data = ori_data[: len(gen_data)]
        anal_sample_no = min([1000, len(ori_data)])
        np.random.seed(0)
        idx = np.random.permutation(min(len(ori_data), len(gen_data)))[:anal_sample_no]
        ori_data = ori_data[idx]
        gen_data = gen_data[idx]
        multi_cat_data = np.concatenate([ori_data, gen_data], axis=0)

        for i in range(min(gen_data.shape[1], 10)):
            cat_data = multi_cat_data[:, i, :]
            tsne = TSNE(
                n_components=2, random_state=0, verbose=1, perplexity=40, n_iter=300
            )
            tsne_obj = tsne.fit_transform(cat_data)

            f, ax = plt.subplots(1)
            plt.scatter(
                tsne_obj[: len(ori_data), 0],
                tsne_obj[: len(ori_data), 1],
                alpha=0.2,
                label="Original(train)", # FIXME: vali にも対応させる？
            )
            plt.scatter(
                tsne_obj[len(ori_data) :, 0],
                tsne_obj[len(ori_data) :, 1],
                alpha=0.2,
                label="Generated",
            )

            ax.legend()
            plt.title(f"t-SNE plot of ch{i} hidden states")
            plt.xlabel("x-tsne")
            plt.ylabel("y-tsne")
            plt.savefig(
                os.path.join(save_dir, f"tsne_hidden_ch{i}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )

            if not self.args.no_wandb:
                wandb.log({f"eval/t-SNE/hidden/ch{i}": wandb.Image(plt)})

    # def plot_hidden_multi_tsne(self, ae_setting, gan_setting):
    #     data_dir = "/home/user/workspace/checkpoints/exp_multi_lr0001_bsz32_iTransformer_ETTm2_ftM_sl432_ll48_pl288_dm128_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_ettm2_projection_0/exp_gan_multi_hidden_gp_gbsz1024_glr0.0001_dlr0.0001_nd128_du5/eval_gan"
    #     save_dir = os.path.join("./checkpoints/", ae_setting, "eval_ae/")
    #     ori_data = np.load(os.path.join(data_dir, "real_reps.npy"))
    #     # gen_data = np.load(os.path.join(save_dir, "hiddens.npy"))

    #     # ori_data = np.squeeze(ori_data)
    #     # ori_data = ori_data[: len(gen_data)]
    #     anal_sample_no = min([1000, len(ori_data)])
    #     np.random.seed(0)
    #     idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    #     ori_data = ori_data[idx]
    #     # gen_data = gen_data[idx]

    #     cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #     f, ax = plt.subplots(1)

    #     multi_cat_data = ori_data.reshape([-1, ori_data.shape[2]])
    #     tsne = TSNE(
    #             n_components=2, random_state=0, verbose=1, perplexity=40, n_iter=300
    #         )
    #     tsne_obj = tsne.fit_transform(multi_cat_data)

    #     for i in range(min(ori_data.shape[1], 10)):
    #         plt.scatter(
    #             tsne_obj[i*1000:(i+1)*1000, 0],
    #             tsne_obj[i*1000:(i+1)*1000, 1],
    #             alpha=0.2,
    #             label=f"ch{i}",
    #             color=cycle[i],
    #         )

    #         ax.legend()
    #     plt.title(f"t-SNE plot of multi hidden states")
    #     plt.xlabel("x-tsne")
    #     plt.ylabel("y-tsne")
    #     plt.savefig(
    #         os.path.join(save_dir, f"tsne_hidden_multi.png"),
    #         bbox_inches="tight",
    #         pad_inches=0,
    #     )

    def save_synth_data(self, ae_setting, gan_setting):
        """
        Synthesize large scale data
        """

        self.model.load_state_dict(
            torch.load(os.path.join("./checkpoints/", ae_setting, "checkpoint.pth"))
        )
        self.generator.load_state_dict(
            torch.load(
                os.path.join(
                    "./checkpoints/",
                    ae_setting,
                    gan_setting,
                    f"generator_iter{self.args.load_iter}.dat",  # {self.args.gan_iter-1}
                )
            )
        )
        self.model.eval()
        self.generator.eval()

        def _gen(batch_size):
            with torch.no_grad():
                z = torch.randn(batch_size, self.args.enc_in, self.args.noise_dim).to(
                    self.device
                )
                x_fake = self.generator(z)  # shape(batch_size, 96)
                # x_fake = torch.unsqueeze(x_fake, dim=1)

                # for anylysis
                # x_fake = np.load('/home/user/workspace/not_outlier_oriandgen_hidden_val_jsai.npy')
                # x_fake = np.expand_dims(x_fake, 1)
                # x_fake = x_fake.astype(np.float32)
                # x_fake = torch.from_numpy(x_fake).clone()
                # x_fake = x_fake.to(self.device)

                dec_out = self.model.decode(x_fake)
                dynamics = dec_out[:, -self.args.pred_len :, :].squeeze().cpu().numpy()

            hidden = []
            res = []

            for i in range(batch_size):
                # dyn = self.dynamic_processor.inverse_transform(dynamics[i]).values.tolist()
                dyn = dynamics[i].tolist()
                res.append(dyn)

                # add hidden_state
                hidden.append(torch.squeeze(x_fake[i]).tolist())
            return res, hidden

        save_dir = os.path.join(
            "./checkpoints/",
            ae_setting,
            gan_setting,
            f"generated_data_iter{self.args.load_iter}",
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # f = h5py.File(f'{save_dir}/ettm2_sl432_pl288.h5', 'w')
        f = h5py.File(f"{save_dir}/data.h5", "w")

        sample_batch_size = 4096
        print("sample size:", self.args.sample_size)
        print("sample batch size: ", sample_batch_size)
        tt = self.args.sample_size // sample_batch_size
        for i in range(tt):
            t1 = time.time()
            data, _ = _gen(sample_batch_size)
            f.create_dataset(f"chunk_{i:05}", data=np.array(data))
            if (i + 1) % 10 == 0:
                print("[Generating -> %d/%d] [time %f]" % (i + 1, tt, time.time() - t1))
        f.close()

    def plot_dec_tsne(self, ae_setting, gan_setting):
        save_dir = os.path.join(
            "./checkpoints/",
            ae_setting,
            gan_setting,
            f"generated_data_iter{self.args.load_iter}/",
        )

        _, train_loader = self._get_data(flag="train")
        _, vali_loader = self._get_data(flag="val")
        train_vali_loader = [train_loader, vali_loader]
        flags = ["train", "val"]

        for i, loader in enumerate(train_vali_loader):
            ori_data = [] # FIXME: 予め保存しておいたデータをloadする
            for j, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                ori_data.append(batch_y.cpu().numpy())
            ori_data = np.vstack(ori_data)
            ori_data = ori_data[:, self.args.label_len :, :]

            # gen_dataset = h5py.File(os.path.join(save_dir, "data.h5"), "r")
            # gen_data = gen_dataset["chunk_00000"]
            # gen_data = np.array(gen_data)
            gen_data = np.load(os.path.join(save_dir, "sample_data.npy"))

            anal_sample_no = min([1000, len(ori_data), len(gen_data)])
            print(f"anal_sample_no: {anal_sample_no}")
            np.random.seed(0)
            ori_idx = np.random.permutation(len(ori_data))[:anal_sample_no]
            gen_idx = np.random.permutation(len(gen_data))[:anal_sample_no]

            ori_data = ori_data[ori_idx]
            gen_data = gen_data[gen_idx]

            assert len(ori_data[0]) == len(gen_data[0])

            multi_cat_data = np.concatenate([ori_data, gen_data], axis=0)

            for k in range(min(gen_data.shape[1], 10)):
                cat_data = multi_cat_data[:, k, :]
                tsne = TSNE(
                    n_components=2, random_state=0, verbose=1, perplexity=40, n_iter=300
                )
                tsne_obj = tsne.fit_transform(cat_data)

                f, ax = plt.subplots(1)
                plt.scatter(
                    tsne_obj[: len(ori_data), 0],
                    tsne_obj[: len(ori_data), 1],
                    alpha=0.2,
                    label=f"Original({flags[i]})",
                )
                plt.scatter(
                    tsne_obj[len(ori_data) :, 0],
                    tsne_obj[len(ori_data) :, 1],
                    alpha=0.2,
                    label="Generated",
                )

                ax.legend()
                plt.title(f"t-SNE plot of generated {k}ch data")
                plt.xlabel("x-tsne")
                plt.ylabel("y-tsne")
                plt.savefig(
                    os.path.join(save_dir, f"tsne_dec_{flags[i]}_ch{k}.png"),
                    bbox_inches="tight",
                    pad_inches=0,
                )

                if not self.args.no_wandb:
                    wandb.log({f"eval/t-SNE/decoded/ch{k}/{flags[i]}": wandb.Image(plt)})
                plt.clf()

    def plot_gen_data(self, ae_setting, gan_setting):
        save_dir = os.path.join(
            "./checkpoints/",
            ae_setting,
            gan_setting,
            f"generated_data_iter{self.args.load_iter}",
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # gen_dataset = h5py.File(os.path.join(save_dir, "data.h5"), "r")
        # gen_data = gen_dataset["chunk_00000"]
        gen_data = np.load(os.path.join(save_dir, "sample_data.npy"))

        for i in range(min(gen_data.shape[2], 10)):
            fig = plt.figure(figsize=(16, 9))
            for j in range(6):
                fig.add_subplot(2, 3, (j + 1))
                plt.plot(gen_data[j, :, i], linewidth=1)  # color='#03af7a'
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
            plt.savefig(
                os.path.join(save_dir, f"list_data_ch{i}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            if not self.args.no_wandb:
                wandb.log({f"eval/generated/ch{i}": wandb.Image(plt)})
            plt.clf()
            plt.close(fig)

    def grad_penalty(self, x_real, x_fake):
        batch_size = x_real.size(0)
        gp_weight = 10

        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        alpha = alpha.expand_as(x_real)

        interpolated = alpha * x_real + (1 - alpha) * x_fake
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        if self.args.use_hidden:
            prob_interpolated = self.discriminator(interpolated)
        else:
            prob_interpolated = self.discriminator(
                torch.permute(interpolated, (0, 2, 1))
            )

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.contiguous().view(batch_size, -1)

        eps = 1e-10
        gradients_norm = torch.sqrt(
            torch.sum(gradients**2, dim=1, dtype=torch.double) + eps
        )

        gradient_penalty = gp_weight * ((gradients_norm - 1) ** 2).mean()
        # print("gradient_penalty: ", gradient_penalty.item())
        return gradient_penalty


# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(),
        inputs=x_in,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert grad_dout2.size() == x_in.size()
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_tgt
        p_tgt.copy_(beta * p_tgt + (1.0 - beta) * p_src)
