python run.py \
--training_ae 0 \
--training_gan 1 \
--root_path ./data/iTransformer_datasets/ETT-small/ \
--data_path ETTm2.csv \
--model_id masked_ae \
--ae_model iTransformer \
--d_model 256 \
--d_ff 512 \
--vari_masked_ratio 0.5 \
--mask_ratio 0.5 \
--data ETTm2 \
--features M \
--seq_len 192 \
--pred_len 192 \
--e_layers 2 \
--enc_in 7 \
--dec_in 7 \
--c_out 7 \
--des ettm2 \
--ae_batch_size 32 \
--train_epochs 30 \
--patience 3 \
--learning_rate 0.001 \
--use_norm 0 \
--gan_model_id mixed_rep_cgan_v2 \
--gan_batch_size 1024 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--noise_dim 256 \
--d_update 10 \
--use_hidden \
--gan_iter 20000 \


# --load_iter 20000 \
# --sample_size 34368 \
# --no_wandb