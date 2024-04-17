#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python run.py --is_ae_training 1 --is_gan_training 0 --is_synthesizig 0 --root_path ./data/informer/ettm/ --data_path ETTm2.csv --model_id ettm2_ae_test --model iTransformer --d_model 128 --data ETTm2 --features S --seq_len 432 --pred_len 288 --e_layers 2 --batch_size 4096 --train_epochs 100 --patience 10 --learning_rate 0.01 --no_wandb

