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
from model.iTransformer import Model

# from model.gan import Discriminator, Generator
from model.SAGAN import Discriminator, Generator
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class Exp_SAGAN(Exp_Basic):
    def __init__(self, args):
        super(Exp_SAGAN, self).__init__(args)

    def _build_model(self):
        model = self.model_dict["SAGAN"].Model(self.args)
        return model

    def _build_generator(self):
        generator = self.model_dict["SAGAN"].Generator(self.args)
        return generator

    def _build_discriminator(self):
        discriminator = self.model_dict["SAGAN"].Discriminator(self.args)
        return discriminator

    def _select_optimizer(self):
        generator_optm = torch.optim.RMSprop(
            params=self.generator.parameters(),
            lr=self.args.disc_lr,
            alpha=self.args.gan_alpha,
        )
        discriminator_optm = torch.optim.RMSprop(
            params=self.discriminator.parameters(),
            lr=self.args.gen_lr,
            alpha=self.args.gan_alpha,
        )
        return generator_optm, discriminator_optm

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def load_ae(self, setting):
        self.ae.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth")))

    def load_generator(self, pretrained_dir=None):
        if pretrained_dir is not None:
            path = pretrained_dir
        else:
            path = "{}/generator.dat".format(self.params["root_dir"])
        self.logger.info("load: " + path)
        self.generator.load_state_dict(torch.load(path, map_location=self.device))

    def train_gan(self, gan_setting):
        _, train_loader = self._get_data(flag="train")

        self.discriminator.train()
        self.generator.train()

        generator_optm, discriminator_optm = self._select_optimizer()

        save_dir = os.path.join("/workspace/checkpoints2/", gan_setting)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        gan_iter = self.args.gan_iter
        d_update = self.args.d_update

        for iteration in range(gan_iter):
            dataloader_iter = iter(train_loader)
            avg_d_loss = 0
            t1 = time.time()

            toggle_grad(self.generator, False)
            toggle_grad(self.discriminator, True)
            self.generator.train()
            self.discriminator.train()

            # train discriminator
            for j in range(d_update):
                for i in range(len(train_loader)):
                    try:
                        batch_x, _, _, _ = next(dataloader_iter)
                    except StopIteration:
                        dataloader_iter = iter(train_loader)
                        batch_x, _, _, _ = next(dataloader_iter)

                    discriminator_optm.zero_grad()
                    z = torch.randn(self.args.gan_batch_size, self.args.noise_dim, 1).to(self.device)

                    d_real = self.discriminator(batch_x.float().to(self.device))

                    # On fake data
                    with torch.no_grad():
                        x_fake = self.generator(z)

                    x_fake.requires_grad_()
                    d_fake = self.discriminator(x_fake)

                    # get gradient penalty
                    gradient_penalty = self.grad_penalty(batch_x.float().to(self.device), x_fake)

                    d_loss = d_fake.mean() - d_real.mean() + gradient_penalty
                    d_loss.backward()

                    discriminator_optm.step()
                    avg_d_loss += (d_fake.mean() - d_real.mean()).item()
                    break

            avg_d_loss /= d_update

            # train generator
            toggle_grad(self.generator, True)
            toggle_grad(self.discriminator, False)
            self.generator.train()
            self.discriminator.train()
            generator_optm.zero_grad()
            z = torch.randn(self.args.gan_batch_size, self.args.noise_dim, 1).to(self.device)
            fake = self.generator(z)

            g_loss = -self.discriminator(fake).mean()
            g_loss.backward()
            generator_optm.step()

            if (iteration + 1) % 100 == 0:
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

            if (iteration + 1) % 1000 == 0:
                gen_data = fake.detach().cpu().numpy()

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
                print(f"Saved Model: {iteration+1}iter ")
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

        print(f"Saved Model: {iteration + 1}iter")
        torch.save(
            self.generator.state_dict(),
            os.path.join(
                save_dir,
                f"generator_iter{iteration + 1}.dat",
            ),
        )

    # def save_hiddens_and_generated_as_npy(self, ae_setting, gan_setting):
    #     """
    #     Save hidden states

    #     Parameters:
    #     ----------
    #     ae_setting : str
    #     gan_setting : str

    #     Returns:
    #     ----------
    #     None : None
    #     """
    #     self.model.load_state_dict(
    #         torch.load(os.path.join("./checkpoints/", ae_setting, "checkpoint.pth"))
    #     )
    #     self.generator.load_state_dict(
    #         torch.load(
    #             os.path.join(
    #                 "./checkpoints/",
    #                 ae_setting,
    #                 gan_setting,
    #                 f"generator_iter{self.args.load_iter}.dat",
    #             )
    #         )
    #     )

    #     print(f"Loading trained {self.args.load_iter} WGAN model")
    #     self.model.eval()
    #     self.generator.eval()

    #     # generate hidden states and save them
    #     with torch.no_grad():
    #         z = torch.randn(
    #             self.args.gan_batch_size, self.args.enc_in, self.args.noise_dim
    #         ).to(self.device)
    #         x_fake = self.generator(z)
    #         hiddens = x_fake.detach().cpu().numpy()
    #         dec_fake = self.model.decode(x_fake)
    #         generated = dec_fake.cpu().numpy()

    #     save_dir = os.path.join("./checkpoints/", ae_setting, gan_setting, "eval_gan/")
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     np.save(
    #         os.path.join(save_dir, "fake_hiddens.npy"), hiddens
    #     )  ### FIXME: shape (batch, noise_dim, enc_in)

    #     save_data_dir = os.path.join(
    #         "./checkpoints/",
    #         ae_setting,
    #         gan_setting,
    #         f"generated_data_iter{self.args.load_iter}",
    #     )
    #     if not os.path.exists(save_data_dir):
    #         os.makedirs(save_data_dir)
    #     np.save(
    #         os.path.join(save_data_dir, "sample_data.npy"), generated
    #     )  # shape (batch, pred_len, N)

    # def plot_hidden_tsne(self, ae_setting, gan_setting):
    #     ori_dir = os.path.join("./checkpoints/", ae_setting, "eval_ae/")
    #     gen_dir = os.path.join("./checkpoints/", ae_setting, gan_setting, "eval_gan/")
    #     ori_data = np.load(
    #         os.path.join(ori_dir, "real_hiddens_train.npy")
    #     )  # FIXME: vali にも対応させる？
    #     gen_data = np.load(os.path.join(gen_dir, "fake_hiddens.npy"))

    #     anal_sample_no = min([1000, len(ori_data), len(gen_data)])
    #     np.random.seed(0)
    #     idx = np.random.permutation(anal_sample_no)[:anal_sample_no]
    #     ori_data = ori_data[idx]
    #     gen_data = gen_data[idx]
    #     multi_cat_data = np.concatenate([ori_data, gen_data], axis=0)

    #     for i in range(min(gen_data.shape[1], 10)):
    #         cat_data = multi_cat_data[:, i, :]
    #         tsne = TSNE(
    #             n_components=2, random_state=0, verbose=1, perplexity=40, n_iter=300
    #         )
    #         tsne_obj = tsne.fit_transform(cat_data)

    #         f, ax = plt.subplots(1)
    #         plt.scatter(
    #             tsne_obj[: len(ori_data), 0],
    #             tsne_obj[: len(ori_data), 1],
    #             alpha=0.2,
    #             label="Original(train)",  # FIXME: vali にも対応させる？
    #         )
    #         plt.scatter(
    #             tsne_obj[len(ori_data) :, 0],
    #             tsne_obj[len(ori_data) :, 1],
    #             alpha=0.2,
    #             label="Generated",
    #         )

    #         ax.legend()
    #         plt.title(f"t-SNE plot of ch{i} hidden states")
    #         plt.xlabel("x-tsne")
    #         plt.ylabel("y-tsne")
    #         plt.savefig(
    #             os.path.join(gen_dir, f"tsne_hidden_ch{i}_train.png"),
    #             bbox_inches="tight",
    #             pad_inches=0,
    #         )
    #         plt.clf()

    #         if not self.args.no_wandb:
    #             wandb.log({f"eval/t-SNE/hidden/ch{i}": wandb.Image(plt)})

    # def plot_dec_tsne(self, ae_setting, gan_setting):
    #     save_dir = os.path.join(
    #         "./checkpoints/",
    #         ae_setting,
    #         gan_setting,
    #         f"generated_data_iter{self.args.load_iter}/",
    #     )

    #     ori_data = np.load(os.path.join("./data/preprocessed_datasets", self.args.des, f"sl{self.args.seq_len}", "prepro_train_shuffled.npy")) # TODO: vari に対応させる
    #     gen_data = np.load(os.path.join(save_dir, "sample_data.npy"))
    #     print(f"ori_data.shape: {ori_data.shape}")
    #     print(f"gen_data.shape: {gen_data.shape}")

    #     anal_sample_no = min([1000, len(ori_data), len(gen_data)])
    #     print(f"anal_sample_no: {anal_sample_no}")
    #     np.random.seed(0)
    #     idx = np.random.permutation(anal_sample_no)[:anal_sample_no]

    #     ori_data = ori_data[idx]
    #     gen_data = gen_data[idx]

    #     assert len(ori_data[0]) == len(gen_data[0])

    #     multi_cat_data = np.concatenate([ori_data, gen_data], axis=0)

    #     for i in range(min(gen_data.shape[2], 10)):
    #         cat_data = multi_cat_data[:, i, :]
    #         tsne = TSNE(
    #             n_components=2, random_state=0, verbose=1, perplexity=40, n_iter=300
    #         )
    #         tsne_obj = tsne.fit_transform(cat_data)

    #         f, ax = plt.subplots(1)
    #         plt.scatter(
    #             tsne_obj[: len(ori_data), 0],
    #             tsne_obj[: len(ori_data), 1],
    #             alpha=0.2,
    #             label=f"Original(train)",
    #         )
    #         plt.scatter(
    #             tsne_obj[len(ori_data) :, 0],
    #             tsne_obj[len(ori_data) :, 1],
    #             alpha=0.2,
    #             label="Generated",
    #         )

    #         ax.legend()
    #         plt.title(f"t-SNE plot of generated {i}ch data")
    #         plt.xlabel("x-tsne")
    #         plt.ylabel("y-tsne")
    #         plt.savefig(
    #             os.path.join(save_dir, f"tsne_dec_ch{i}_train.png"),
    #             bbox_inches="tight",
    #             pad_inches=0,
    #         )

    #         if not self.args.no_wandb:
    #             wandb.log(
    #                 {f"eval/t-SNE/decoded/ch{i}": wandb.Image(plt)}
    #             )
    #         plt.clf()

    # def plot_gen_data(self, ae_setting, gan_setting):
    #     save_dir = os.path.join(
    #         "./checkpoints/",
    #         ae_setting,
    #         gan_setting,
    #         f"generated_data_iter{self.args.load_iter}",
    #     )
    #     gen_data = np.load(os.path.join(save_dir, "sample_data.npy"))

    #     for i in range(min(gen_data.shape[2], 10)):
    #         fig = plt.figure(figsize=(20, 20))
    #         for j in range(9):
    #             fig.add_subplot(3, 3, (j + 1))
    #             plt.plot(gen_data[j, :, i], linewidth=1)  # color='#03af7a'
    #             plt.xticks(fontsize=10)
    #             plt.yticks(fontsize=10)
    #         plt.savefig(
    #             os.path.join(save_dir, f"list_gendata_ch{i}.png"),
    #             bbox_inches="tight",
    #             pad_inches=0,
    #         )
    #         if not self.args.no_wandb:
    #             wandb.log({f"eval/generated/ch{i}": wandb.Image(plt)})
    #         plt.clf()
    #         plt.close(fig)

    # def save_synth_data_as_h5(self, ae_setting, gan_setting):
    #     """
    #     Synthesize large scale data for data augmentation
    #     """

    #     self.model.load_state_dict(
    #         torch.load(os.path.join("./checkpoints/", ae_setting, "checkpoint.pth"))
    #     )
    #     self.generator.load_state_dict(
    #         torch.load(
    #             os.path.join(
    #                 "./checkpoints/",
    #                 ae_setting,
    #                 gan_setting,
    #                 f"generator_iter{self.args.load_iter}.dat",
    #             )
    #         )
    #     )
    #     self.model.eval()
    #     self.generator.eval()

    #     def _gen(batch_size):
    #         with torch.no_grad():
    #             z = torch.randn(batch_size, self.args.enc_in, self.args.noise_dim).to(
    #                 self.device
    #             )
    #             x_fake = self.generator(z)  # shape(batch_size, 96)
    #             # x_fake = torch.unsqueeze(x_fake, dim=1)

    #             # for anylysis
    #             # x_fake = np.load('/home/user/workspace/not_outlier_oriandgen_hidden_val_jsai.npy')
    #             # x_fake = np.expand_dims(x_fake, 1)
    #             # x_fake = x_fake.astype(np.float32)
    #             # x_fake = torch.from_numpy(x_fake).clone()
    #             # x_fake = x_fake.to(self.device)

    #             dec_out = self.model.decode(x_fake)
    #             dynamics = dec_out[:, -self.args.pred_len :, :].squeeze().cpu().numpy()

    #         res = []

    #         for i in range(batch_size):
    #             # dyn = self.dynamic_processor.inverse_transform(dynamics[i]).values.tolist()
    #             dyn = dynamics[i].tolist()
    #             res.append(dyn)

    #             # add hidden_state
    #             # hidden.append(torch.squeeze(x_fake[i]).tolist())
    #         return res

    #     save_dir = os.path.join(
    #         "./checkpoints/",
    #         ae_setting,
    #         gan_setting,
    #         f"generated_data_iter{self.args.load_iter}",
    #     )
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     # f = h5py.File(f'{save_dir}/ettm2_sl432_pl288.h5', 'w')
    #     f = h5py.File(f"{save_dir}/gen.h5", "w")

    #     sample_batch_size = 4096
    #     print("sample size:", self.args.sample_size)
    #     print("sample batch size: ", sample_batch_size)
    #     tt = self.args.sample_size // sample_batch_size
    #     for i in range(tt):
    #         t1 = time.time()
    #         data = _gen(sample_batch_size)
    #         f.create_dataset(f"chunk_{i:05}", data=np.array(data))
    #         if (i + 1) % 10 == 0:
    #             print("[Generating -> %d/%d] [time %f]" % (i + 1, tt, time.time() - t1))
    #     f.close()

    # def save_synth_data_as_npy(self, ae_setting, gan_setting):
    #     """
    #     Synthesize large scale data for data augmentation
    #     """

    #     self.model.load_state_dict(
    #         torch.load(os.path.join("./checkpoints/", ae_setting, "checkpoint.pth"))
    #     )
    #     self.generator.load_state_dict(
    #         torch.load(
    #             os.path.join(
    #                 "./checkpoints/",
    #                 ae_setting,
    #                 gan_setting,
    #                 f"generator_iter{self.args.load_iter}.dat",
    #             )
    #         )
    #     )
    #     self.model.eval()
    #     self.generator.eval()

    #     def _gen(batch_size):
    #         with torch.no_grad():
    #             z = torch.randn(batch_size, self.args.enc_in, self.args.noise_dim).to(
    #                 self.device
    #             )
    #             x_fake = self.generator(z)
    #             dec_out = self.model.decode(x_fake)
    #             dynamics = dec_out[:, -self.args.pred_len :, :].squeeze().cpu().numpy()
    #         res = []
    #         for i in range(batch_size):
    #             #dyn = self.dynamic_processor.inverse_transform(dynamics[i]).values.tolist()
    #             dyn = dynamics[i].tolist()
    #             res.append(dyn)
    #         return res

    #     data = []
    #     # tt = n // batch_size
    #     sample_batch_size = 4096
    #     print("sample size:", self.args.sample_size)
    #     print("sample batch size: ", sample_batch_size)
    #     tt = self.args.sample_size // sample_batch_size
    #     for i in range(tt):
    #         data.extend(_gen(sample_batch_size))
    #     res = self.args.sample_size - tt * sample_batch_size
    #     if res>0:
    #         data.extend(_gen(res))

    #     save_dir = os.path.join(
    #         "./checkpoints/",
    #         ae_setting,
    #         gan_setting,
    #         f"generated_data_iter{self.args.load_iter}",
    #     )
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     np.save(os.path.join(save_dir, "gen_large.npy"), np.array(data))

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
            prob_interpolated = self.discriminator(torch.permute(interpolated, (0, 2, 1)))

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
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1, dtype=torch.double) + eps)

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
