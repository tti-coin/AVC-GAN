"""
train and evaluate AE
"""

import argparse
import os
import pdb
import random

import numpy as np
import torch

from experiments.exp_auto_encoder import Exp_AutoEncoder
from experiments.exp_conditional_gan import Exp_iTransGAN
from experiments.exp_long_term_forecasting_partial import Exp_Long_Term_Forecast_Partial

if __name__ == "__main__":
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="iTransformer")

    # basic experiment setting
    parser.add_argument("--training_ae", type=int, default=1, help="status")
    parser.add_argument("--training_gan", type=int, default=0, help="status")
    parser.add_argument("--evaluating_gan", type=int, default=0, help="status")
    parser.add_argument("--synthesizing", type=int, default=0, help="status")

    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--ae_model",
        type=str,
        required=True,
        default="iTransformer",
        help="model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]",
    )

    # data loader
    parser.add_argument("--pin_memory", type=bool, default=True, help="pin memory")
    parser.add_argument(
        "--data", type=str, required=True, default="custom", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/electricity/",
        help="root path of the data file",
    )
    parser.add_argument(
        "--data_path", type=str, default="electricity.csv", help="data csv file"
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument(
        "--label_len", type=int, default=48, help="start token length"
    )  # no longer needed in inverted Transformers
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )

    #### AutoEncoder setting
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument(
        "--c_out", type=int, default=7, help="output size"
    )  # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    # added for MLM
    parser.add_argument("--vari_masked_ratio", type=float, default=0.5)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in ecoder",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="whether to predict unseen future data",
    )
    parser.add_argument(
        "--eval_reconstruct",
        action="store_true",
        help="whether to reconstruct input data",
    )
    parser.add_argument(
        "--do_synthesize", action="store_true", help="whether to use synthetic data"
    )
    parser.add_argument("--fix_gan", default=None, help="test mode for GAN")

    ### GAN setting
    parser.add_argument("--gan_model_id", type=str, default="test", help="gan model id")
    parser.add_argument(
        "--gan_batch_size", type=int, default=1024, help="batch size of WGAN"
    )
    parser.add_argument(
        "--gen_lr", type=float, default=1e-4, help="WGAN generator learning rate"
    )
    parser.add_argument(
        "--disc_lr", type=float, default=1e-4, help="WGAN discriminator learning rate"
    )
    parser.add_argument("--gan_alpha", type=float, default=0.99, help="for RMSprop")
    parser.add_argument(
        "--noise_dim", type=int, default=128, help="dim of WGAN noise state"
    )
    parser.add_argument(
        "--gan_iter",
        type=int,
        default=20000,
        help="Number of iterations through training set for WGAN",
    )
    parser.add_argument(
        "--d_update",
        type=int,
        default=5,
        help="discriminator updates per generator update",
    )
    parser.add_argument(
        "--use_hidden", action="store_true", help="use hidden states for GAN training"
    )

    # synthtesizing
    parser.add_argument(
        "--load_iter", type=int, default=19999, help="which iteration of model to use"
    )
    parser.add_argument(
        "--sample_size", type=int, default=2048, help="sample size of synthetic data"
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--ae_batch_size",
        type=int,
        default=32,
        help="batch size of train input data for AE",
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    )

    # iTransformer
    parser.add_argument(
        "--exp_name",
        type=str,
        required=False,
        default="MTSF",
        help="experiemnt name, options:[MTSF, partial_train]",
    )
    parser.add_argument(
        "--channel_independence",
        type=bool,
        default=False,
        help="whether to use channel_independence mechanism",
    )
    parser.add_argument(
        "--inverse", action="store_true", help="inverse output data", default=False
    )
    parser.add_argument(
        "--class_strategy",
        type=str,
        default="projection",
        help="projection/average/cls_token",
    )
    parser.add_argument(
        "--target_root_path",
        type=str,
        default="./data/electricity/",
        help="root path of the data file",
    )
    parser.add_argument(
        "--target_data_path", type=str, default="electricity.csv", help="data file"
    )
    parser.add_argument(
        "--efficient_training",
        type=bool,
        default=False,
        help="whether to use efficient_training (exp_name should be partial train)",
    )  # See Figure 8 of our paper for the detail
    parser.add_argument(
        "--use_norm", type=int, default=False, help="use norm and denorm"
    )
    parser.add_argument(
        "--partial_start_index",
        type=int,
        default=0,
        help="the start index of variates for partial training, "
        "you can select [partial_start_index, min(enc_in + partial_start_index, N)]",
    )

    # wandb
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb")
    parser.add_argument(
        "--wandb_notes", type=str, default="test", help="desctiption of experiment"
    )
    # parser.add_argument("--wandb_entity", type=str, default="test", help="wandb entity")
    # parser.add_argument("--wandb_project", type=str, default="test", help="wandb project")
    parser.add_argument(
        "--wandb_run_name", type=str, default="test", help="wandb run name"
    )
    # parser.add_argument("--wandb_group", type=str, default="test", help="wandb group")
    # parser.add_argument("--wandb_tags", type=str, default="test", help="wandb tags")

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    args_dict = vars(args)
    args.model_id = f"{args.model_id}_{args.des}_sl{args.seq_len}_pl{args.pred_len}"
    args.gan_model_id = f"{args.gan_model_id}_{args.des}_pl{args.pred_len}_vmr{args.vari_masked_ratio}_mr{args.mask_ratio}_hd{args.use_hidden}_du{args.d_update}"

    if args.training_ae:
        # if args.exp_name == "partial_train":  # See Figure 8 of our paper, for the detail
        #     Exp = Exp_Long_Term_Forecast_Partial
        # MTSF: multivariate time series forecasting
        Exp = Exp_AutoEncoder

        # if args.is_ae_training:
        for ii in range(args.itr):
            ae_setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_vmr{}_mr{}_dt{}_lr{}_{}_{}_{}".format(
                args.model_id,
                args.ae_model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.vari_masked_ratio,
                args.mask_ratio,
                args.distil,
                args.learning_rate,
                args.des,
                args.class_strategy,
                ii,
            )

            exp = Exp(args)  # set experiments

            if not args.no_wandb and ii == 0:
                import wandb

                if not os.path.exists(os.path.join("/workspace/logs/" + ae_setting)):
                    os.makedirs(os.path.join("/workspace/logs/" + ae_setting))
                wandb.init(
                    project="Masked-iTransGAN",
                    config=args,
                    tags=["AE"],
                    name=args.model_id,
                    notes=args.wandb_notes,
                    dir=os.path.join("/workspace/logs/" + ae_setting),
                    # resume="allow",
                    # id=args.wandb_id,
                )
            else:
                wandb = None

            print("========== Start AE training ==========")
            exp.train(ae_setting)

            print("========== Start AE evaluation ==========")
            print("Reconstructing from training and validation data")
            exp.save_recon_as_npy(ae_setting, True)

            print("Plotting original and reconstructed data")
            exp.plot_recon_as_tsne(ae_setting)
            # exp.plot_multi_hidden_as_tsne(ae_setting)
            # pdb.set_trace()

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                    ae_setting
                )
            )
            exp.test(ae_setting, test=1)
            torch.cuda.empty_cache()

    if args.training_gan:
        assert args.gan_model_id is not None
        Exp = Exp_iTransGAN
        args.exp_name = "gan"

        for ii in range(args.itr):
            ae_setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_vmr{}_mr{}_dt{}_lr{}_{}_{}_{}".format(
                args.model_id,
                args.ae_model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.vari_masked_ratio,
                args.mask_ratio,
                args.distil,
                args.learning_rate,
                args.des,
                args.class_strategy,
                ii,
            )

            gan_setting = "{}_gbsz{}_glr{}_dlr{}_nd{}_du{}_hd{}".format(
                args.gan_model_id,
                args.gan_batch_size,
                args.gen_lr,
                args.disc_lr,
                args.noise_dim,
                args.d_update,
                args.use_hidden,
            )

            exp = Exp(args)  # set experiments

            if not args.no_wandb:
                import wandb

                if not os.path.exists(
                    os.path.join("/workspace/logs/gan" + gan_setting)
                ):
                    os.makedirs(os.path.join("/workspace/logs/gan" + gan_setting))
                wandb.init(
                    project="Masked-iTransGAN",
                    config=args,
                    tags=["GAN"],
                    name=args.gan_model_id,
                    notes=args.wandb_notes,
                    dir=os.path.join("/workspace/logs/" + gan_setting),
                )
            else:
                wandb = None

            print("Start training iTransGAN")
            exp.train_gan(ae_setting, gan_setting)

            torch.cuda.empty_cache()

    if args.evaluating_gan:
        assert args.gan_model_id is not None
        Exp = Exp_iTransGAN
        args.exp_name = "gan"

        for ii in range(args.itr):
            ae_setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_vmr{}_mr{}_dt{}_lr{}_{}_{}_{}".format(
                args.model_id,
                args.ae_model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.vari_masked_ratio,
                args.mask_ratio,
                args.distil,
                args.learning_rate,
                args.des,
                args.class_strategy,
                ii,
            )

            gan_setting = "{}_gbsz{}_glr{}_dlr{}_nd{}_du{}_hd{}".format(
                args.gan_model_id,
                args.gan_batch_size,
                args.gen_lr,
                args.disc_lr,
                args.noise_dim,
                args.d_update,
                args.use_hidden,
            )

            exp = Exp(args)

        print("Start iTransGAN evaluatining")
        exp.save_hiddens_and_generated_as_npy(ae_setting, gan_setting)

        print("plotting some hidden states and real representations with t-SNE")
        exp.plot_hidden_tsne(ae_setting, gan_setting)

        print("Plotting generated data with t-SNE and matplotlib")
        exp.plot_dec_tsne(ae_setting, gan_setting)
        exp.plot_gen_data(ae_setting, gan_setting)

        torch.cuda.empty_cache()

    if args.synthesizing:
        assert args.gan_model_id is not None
        Exp = Exp_iTransGAN
        args.exp_name = "gan"

        ii = 0
        ae_setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_vmr{}_mr{}_dt{}_lr{}_{}_{}_{}".format(
            args.model_id,
            args.ae_model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.vari_masked_ratio,
            args.mask_ratio,
            args.distil,
            args.learning_rate,
            args.des,
            args.class_strategy,
            ii,
        )

        gan_setting = "{}_gbsz{}_glr{}_dlr{}_nd{}_du{}_hd{}".format(
            args.gan_model_id,
            args.gan_batch_size,
            args.gen_lr,
            args.disc_lr,
            args.noise_dim,
            args.d_update,
            args.use_hidden,
        )

        exp = Exp(args)  # set experiments

        print("Start synthesizing data and save them to npy files")
        # exp.save_synth_data_as_h5(ae_setting, gan_setting)
        exp.save_synth_data_as_npy(ae_setting, gan_setting)

        torch.cuda.empty_cache()
