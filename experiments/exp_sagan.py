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
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings("ignore")


class Exp_SAGAN(Exp_Basic):
    def __init__(self, args):
        super(Exp_SAGAN, self).__init__(args)

    def _build_model(self):
        model = (
            self.model_dict[self.args.ae_model].Model(self.args, self.device).float()
        )
        return model

    def _build_generator(self):
        generator = (
            self.model_dict[self.args.gan_model]
            .Generator(self.args, self.device)
            .float()
        )
        print(generator)
        return generator

    def _build_discriminator(self):
        discriminator = self.model_dict[self.args.gan_model].Discriminator(
            self.args, self.device
        )
        return discriminator

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        generator_optim = torch.optim.RMSprop(
            params=self.generator.parameters(),
            lr=self.args.disc_lr,
            alpha=self.args.gan_alpha,
        )
        discriminator_optim = torch.optim.RMSprop(
            params=self.discriminator.parameters(),
            lr=self.args.gen_lr,
            alpha=self.args.gan_alpha,
        )
        return model_optim, generator_optim, discriminator_optim

    def train_gan(self, ae_setting, gan_setting):
        self.model.load_state_dict(
            torch.load(
                os.path.join("/workspace/checkpoints/", ae_setting, "checkpoint.pth")
            )
        )
        print("Loaded trained AutoEncoder")

        _, train_loader = self._get_data(flag="train")
        dataloader_iter = iter(train_loader)

        self.generator.train()
        self.discriminator.train()
        self.model.eval()

        _, generator_optim, discriminator_optim = self._select_optimizer()

        save_dir = os.path.join("/workspace/checkpoints/", ae_setting, gan_setting)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        gan_iter = self.args.gan_iter
        d_update = self.args.d_update
        accumulation_steps = self.args.accumulation_steps

        initial_noise = None

        for iteration in range(gan_iter):
            avg_d_loss = 0
            t1 = time.time()

            toggle_grad(self.generator, False)
            toggle_grad(self.discriminator, True)
            self.generator.train()
            self.discriminator.train()

            # train discriminator
            for j in range(d_update):
                for acc in range(accumulation_steps):
                    try:
                        batch_x, _, _, _ = next(dataloader_iter)
                    except StopIteration:
                        dataloader_iter = iter(train_loader)
                        batch_x, _, _, _ = next(dataloader_iter)

                    # discriminator_optim.zero_grad()

                    if initial_noise is None:
                        initial_noise = torch.randn(
                            self.args.gan_batch_size,
                            self.args.enc_in,
                            self.args.d_model,
                            device=self.device,
                        )
                        noise = initial_noise
                    else:
                        noise = torch.randn_like(initial_noise)

                    z_real = self.model.encode(batch_x.float().to(self.device))
                    d_real = self.discriminator(z_real)

                    # On fake data
                    with torch.no_grad():
                        z_fake = self.generator(noise)

                    z_fake.requires_grad_()
                    d_fake = self.discriminator(z_fake)

                    # get gradient penalty
                    gradient_penalty = self.grad_penalty(z_real, z_fake)
                    d_loss = d_fake.mean() - d_real.mean() + gradient_penalty

                    # accumulation_steps = 2
                    d_loss = d_loss / accumulation_steps
                    d_loss.backward()

                    avg_d_loss += (d_fake.mean() - d_real.mean()).item()
                discriminator_optim.step()
                discriminator_optim.zero_grad()


            avg_d_loss /=  d_update

            # train generator
            toggle_grad(self.generator, True)
            toggle_grad(self.discriminator, False)
            self.generator.train()
            self.discriminator.train()
            # generator_optim.zero_grad()

            for acc in range(accumulation_steps):
                noise = torch.randn_like(initial_noise)
                fake = self.generator(noise)
                g_loss = -self.discriminator(fake).mean()

                g_loss = g_loss / accumulation_steps
                g_loss.backward()

            generator_optim.step()
            generator_optim.zero_grad()
            # generator_optim.step()

            if (iteration + 1) % 1000 == 0:
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

            # if (iteration + 1) % 10000 == 0:
            #     print(f"Save {iteration+1}iter model")
            #     torch.save(
            #         self.generator.state_dict(),
            #         os.path.join(
            #             save_dir,
            #             f"generator_iter{iteration+1}.dat",
            #         ),
            #     )

        torch.save(
            self.generator.state_dict(),
            os.path.join(
                save_dir,
                f"generator_iter{iteration + 1}.dat",
            ),
        )
        torch.save(
            self.discriminator.state_dict(),
            os.path.join(
                save_dir,
                f"disc_iter{iteration + 1}.dat",
            ),
        )
        print(f"Saved {iteration + 1}iter Conditional-WGAN")


    def plot_gen_data(self, ae_setting, gan_setting):
        save_dir = os.path.join(
            "/workspace/checkpoints/",
            ae_setting,
            gan_setting,
            f"generated_data_iter{self.args.load_iter}",
        )
        gen_data = np.load(os.path.join(save_dir, "sample_data.npy"))

        for i in range(min(gen_data.shape[2], 10)):
            fig = plt.figure(figsize=(20, 20))
            for j in range(9):
                fig.add_subplot(3, 3, (j + 1))
                plt.plot(gen_data[j, :, i], linewidth=1)  # color='#03af7a'
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
            plt.savefig(
                os.path.join(save_dir, f"list_gendata_ch{i}.png"),
                bbox_inches="tight",
                pad_inches=0,
            )
            if not self.args.no_wandb:
                wandb.log({f"eval/generated/ch{i}": wandb.Image(plt)})
            plt.clf()
            plt.close(fig)

    def save_synth_data_as_npy(self, ae_setting, gan_setting):
        """
        Synthesize large scale data for data augmentation
        """

        self.model.load_state_dict(
            torch.load(
                os.path.join("/workspace/checkpoints/", ae_setting, "checkpoint.pth")
            )
        )
        self.generator.load_state_dict(
            torch.load(
                os.path.join(
                    "/workspace/checkpoints/",
                    ae_setting,
                    gan_setting,
                    f"generator_iter{self.args.load_iter}.dat",
                )
            )
        )
        self.model.eval()
        self.generator.eval()

        def _gen(batch_size):
            with torch.no_grad():
                noise = torch.randn(batch_size, self.args.enc_in, self.args.d_model).to(
                    self.device
                )
                z_fake = self.generator(noise)
                dec_out = self.model.decode(z_fake)
                dynamics = dec_out[:, -self.args.pred_len :, :].squeeze().cpu().numpy()
            res = []
            for i in range(batch_size):
                # dyn = self.dynamic_processor.inverse_transform(dynamics[i]).values.tolist()
                dyn = dynamics[i].tolist()
                res.append(dyn)
            return res

        data = []
        # tt = n // batch_size
        sample_batch_size = 4096
        print("sample size:", self.args.sample_size)
        print("sample batch size: ", sample_batch_size)
        tt = self.args.sample_size // sample_batch_size
        for i in range(tt):
            data.extend(_gen(sample_batch_size))
        res = self.args.sample_size - tt * sample_batch_size
        if res > 0:
            data.extend(_gen(res))

        save_dir = os.path.join(
            "/workspace/checkpoints/",
            ae_setting,
            gan_setting,
            f"generated_data_iter{self.args.load_iter}",
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, "gen.npy"), np.array(data))

    def grad_penalty(self, z_real, z_fake):
        batch_size = z_real.size(0)
        gp_weight = 10

        if z_real.dim() == 2:
            alpha = torch.rand(batch_size, 1, device=self.device)
        elif z_real.dim() == 3:
            alpha = torch.rand(batch_size, 1, 1, device=self.device)
        alpha = alpha.expand_as(z_real)

        interpolated = (alpha * z_real + (1 - alpha) * z_fake).requires_grad_(True)

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            # grad_outputs=torch.ones(prob_interpolated.size(), device=self.device),
            grad_outputs=torch.ones_like(prob_interpolated, device=self.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.contiguous().view(batch_size, -1)

        eps = 1e-10
        gradients_norm = torch.sqrt(
            torch.sum(gradients**2, dim=1, dtype=torch.double) + eps
        )

        gradient_penalty = gp_weight * ((gradients_norm - 1) ** 2).mean()
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
