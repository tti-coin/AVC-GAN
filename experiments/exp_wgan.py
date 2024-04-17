"""
潜在表現で識別器を学習する実験クラス
Gradient Penalty部分の実装を変更
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
        self.generator = Generator(self.args.noise_dim).to(self.device)
        print(self.generator)

        self.discriminator = Discriminator(self.args.noise_dim).to(self.device)
        # self.discriminator = Discriminator(self.args.pred_len).to(self.device) # TODO
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
        self.model.train()
        print("Traning GAN model with AE.train() mode")
        # self.model.eval()
        # print("Change AE model to eval mode during GAN training")

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
                    self.discriminator_optm.zero_grad()
                    z = torch.randn(self.args.gan_batch_size, self.args.noise_dim).to(
                        self.device
                    )

                    real_rep = self.model.encode(batch_x.float().to(self.device))
                    # real_dec = self.model.decode(real_rep)
                    d_real = self.discriminator(torch.squeeze(real_rep))

                    # On fake data
                    with torch.no_grad():
                        x_fake = self.generator(z)

                    x_fake.requires_grad_()
                    x_fake = torch.unsqueeze(x_fake, dim=1)
                    # fake_dec = self.model.decode(x_fake)
                    # fake_dec = torch.squeeze(fake_dec)
                    d_fake = self.discriminator(x_fake)

                    # get gradient penalty
                    # reg = 10 * self.wgan_gp_reg(real_rep, x_fake)
                    # real_dec = torch.squeeze(real_dec)
                    # reg = 10 * self.wgan_gp_reg(real_dec, fake_dec)
                    gradient_penalty = self.grad_penalty(real_rep, x_fake)

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
            z = torch.randn(self.args.gan_batch_size, self.args.noise_dim).to(
                self.device
            )
            fake = self.generator(z)
            fake = torch.unsqueeze(fake, dim=1)
            # fake_data = self.model.decode(fake)
            # fake_data = torch.squeeze(fake_data)
            g_loss = -self.discriminator(fake).mean()
            g_loss.backward()
            self.generator_optm.step()

            if iteration % 1 == 0:
                print(
                    "[Iteration: %d/%d] [Time: %f] [D_loss: %f] [G_loss: %f] [gp: %f]"
                    % (
                        iteration,
                        gan_iter,
                        time.time() - t1,
                        avg_d_loss,
                        g_loss.item(),
                        gradient_penalty.item(),
                    )
                )

            if not self.args.no_wandb:
                log_dict = {
                    "train/d_loss": avg_d_loss,
                    "train/d_loss_real": d_real.mean(),
                    "train/d_loss_fake": d_fake.mean(),
                    "train/g_loss": g_loss.item(),
                    "train/gradient_penalty": gradient_penalty.item(),
                }
                wandb.log(log_dict)

            if (iteration + 1) % 5000 == 0:
                print(f"Save {iteration}iter WGAN model")
                torch.save(
                    self.generator.state_dict(),
                    os.path.join(
                        save_dir,
                        f"generator_iter{iteration}.dat",
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

        print(f"Save {iteration}iter WGAN model")
        torch.save(
            self.generator.state_dict(),
            os.path.join(
                save_dir,
                f"generator_iter{iteration}.dat",
            ),
        )

    def generate_hiddens_real_reps(self, ae_setting, gan_setting):
        """
        Save hidden states and real representations

        Parameters:
        ----------
        ae_setting : str
        gan_setting : str

        Returns:
        ----------
        None
            Save hidden states and real representations in the specified directory
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
                    f"generator_iter{self.args.load_iter}.dat",
                )
            )
        )
        print(f"Loading trained {self.args.load_iter} WGAN model")
        self.model.eval()
        self.generator.eval()

        def _gen(batch_size):
            with torch.no_grad():
                z = torch.randn(batch_size, self.args.noise_dim).to(self.device)
                x_fake = self.generator(z)
                x_fake = torch.unsqueeze(x_fake, dim=1)

                dec_out = self.model.decode(x_fake)
                dynamics = dec_out[:, -self.args.pred_len :, :].squeeze().cpu().numpy()

            hidden = []
            res = []

            for i in range(batch_size):
                dyn = dynamics[i].tolist()
                res.append(dyn)
                hidden.append(torch.squeeze(x_fake[i]).tolist())

            return res, hidden

        save_dir = os.path.join("./checkpoints/", ae_setting, gan_setting, "eval_gan/")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        res, hidden = _gen(self.args.gan_batch_size)
        # pdb.set_trace()
        np.save(os.path.join(save_dir, "hiddens.npy"), hidden)

        # evaluate real representation
        # val_data, val_loader = self._get_data(flag="val")

        train_data, val_loader = self._get_data(flag="train")

        recons = []
        trues = []
        real_reps = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                val_loader
            ):
                # enc_out, attns = self.encode(enc_out, attn_mask=None)
                enc_out = self.model.encode(batch_x.float().to(self.device))

                real_rep = enc_out.detach().cpu().numpy()

                dec_out = self.model.decode(enc_out)
                recon = dec_out.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                recons.append(recon)
                trues.append(true)
                real_reps.append(real_rep)

        recons = np.array(recons)
        trues = np.array(trues)
        real_reps = np.array(real_reps)
        recons = recons.reshape(-1, recons.shape[-2], recons.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        real_reps = real_reps.reshape(-1, real_reps.shape[-2], real_reps.shape[-1])

        np.save(os.path.join(save_dir, "real_reps.npy"), real_reps)


    def plot_hidden_tsne(self, ae_setting, gan_setting):
        save_dir = os.path.join("./checkpoints/", ae_setting, gan_setting, "eval_gan/")
        ori_data = np.load(os.path.join(save_dir, "real_reps.npy"))
        gen_data = np.load(os.path.join(save_dir, "hiddens.npy"))

        ori_data = np.squeeze(ori_data)
        ori_data = ori_data[: len(gen_data)]
        anal_sample_no = min([1000, len(ori_data)])
        np.random.seed(0)
        idx = np.random.permutation(len(ori_data))[:anal_sample_no]

        ori_data = ori_data[idx]
        recon_data = recon_data[idx]

        cat_data = np.concatenate([ori_data, gen_data], axis=0)

        tsne = TSNE(
            n_components=2, random_state=0, verbose=1, perplexity=40, n_iter=300
        )
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

        if not self.args.no_wandb:
            wandb.log({f"t-SNE/hidden": wandb.Image(plt)})

    def save_synth_data(self, ae_setting, gan_setting):
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
                z = torch.randn(batch_size, self.args.noise_dim).to(self.device)
                x_fake = self.generator(z)  # shape(batch_size, 96)
                x_fake = torch.unsqueeze(x_fake, dim=1)

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
            "./checkpoints/", ae_setting, gan_setting, "generated_data"
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
            "./checkpoints/", ae_setting, gan_setting, f"generated_data_iter{self.args.load_iter}/"
        )

        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        ori_data = []
        # cnt = 1000 // self.args.ae_batch_size + 1
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            ori_data.append(batch_y.cpu().numpy())
        ori_data = np.vstack(ori_data)
        ori_data = ori_data[:,self.args.label_len:,:]
        # TODO: ノイズから生成した表現をデコードしたものと同等のものを取り出したい

        # ori_data = pickle.load(open("/workspace/data/preprocessed_for_eval/ori_ettm2-288-jsai.pkl", "rb"))
        # ori_data = ori_data.squeeze()

        gen_dataset = h5py.File(os.path.join(save_dir, "data.h5"), "r")
        gen_data = gen_dataset["chunk_00000"]
        gen_data = np.array(gen_data)

        ori_data = np.squeeze(ori_data)
        anal_sample_no = min([1000, len(ori_data)])
        np.random.seed(0)
        ori_idx = np.random.permutation(len(ori_data))[:anal_sample_no]
        gen_idx = np.random.permutation(len(gen_data))[:anal_sample_no]

        ori_data = ori_data[ori_idx]
        gen_data = gen_data[gen_idx]

        assert len(ori_data[0]) == len(gen_data[0])

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
        plt.savefig(os.path.join(save_dir, "tsne_dec_new.png"))
        

    def grad_penalty(self, x_real, x_fake):
        batch_size = x_real.size(0)
        gp_weight = 10

        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        alpha = alpha.expand_as(x_real)

        interpolated = alpha * x_real + (1 - alpha) * x_fake
        interpolated = Variable(interpolated, requires_grad=True).to(self.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(batch_size, -1)

        eps = 1e-10
        gradients_norm = torch.sqrt(
            torch.sum(gradients**2, dim=1, dtype=torch.double) + eps
        )

        gradient_penalty = gp_weight * ((gradients_norm - 1) ** 2).mean()
        # print("gradient_penalty: ", gradient_penalty.item())
        return gradient_penalty

    # def grad_penalty(self, x_real, x_fake, center=1.0):
    #     batch_size = x_real.size(0)
    #     # print(f'x_real:{x_real.shape}, x_fake:{x_fake.shape}')
    #     eps = torch.rand(batch_size, device=self.device).view(batch_size, -1)
    #     x_interp = (
    #         1 - eps
    #     ) * x_real + eps * x_fake  # interpolation between real and fake data
    #     x_interp = x_interp.detach()
    #     x_interp.requires_grad_()
    #     d_out = self.discriminator(x_interp)

    #     # reg: 勾配の2乗の平方根から中心を引いて2乗したものの平均（勾配が中心に対してどれだけ離れているかを表すペナルティ）
    #     reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()
    #     print("gradient_penalty (reg): ", reg.item())
    #     return reg


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
