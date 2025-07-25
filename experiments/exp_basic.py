import os
import pdb

import torch

from model import SAGAN, ConditionalSAGAN, iTransformer


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "iTransformer": iTransformer,
            "SAGAN": SAGAN,
            "ConditionalSAGAN": ConditionalSAGAN,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        if args.exp_name == "GAN":
            self.generator = self._build_generator().to(self.device)
            self.discriminator = self._build_discriminator().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _build_generator(self):
        raise NotImplementedError
        return None

    def _build_discriminator(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            # print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device("cpu")
            # print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
