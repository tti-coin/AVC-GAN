# Training
python run_cgan.py \
--is_gan_training 1 \
--is_gan_evaluating 0 \
--root_path ./data/iTransformer_datasets/ETT-small/ \
--data_path ETTh1.csv \
--model_id ae \
--model iTransformer \
--d_model 128 \
--data ETTh1 \
--features M \
--seq_len 192 \
--pred_len 192 \
--e_layers 2 \
--enc_in 7 \
--dec_in 7 \
--c_out 7 \
--des etth1 \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001 \
--gan_model_id test3-cgan \
--use_hidden \
--gan_batch_size 1024 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--noise_dim 128 \
--d_update 10 \
--gan_iter 20000 \
--no_wandb
# --wandb_notes "Performance check after refactoring (not changed gradient penalty)"


# python run_cgan.py \
# --is_gan_training 0 \
# --is_gan_evaluating 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTh1.csv \
# --model_id ae \
# --model iTransformer \
# --d_model 128 \
# --data ETTh1 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des etth1 \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --gan_model_id test-cgan \
# --use_hidden \
# --gan_batch_size 1024 \
# --gen_lr 0.0001 \
# --disc_lr 0.0001 \
# --noise_dim 128 \
# --d_update 10 \
# --gan_iter 20000 \
# --sample_size 4096 \
# --load_iter 20000 \
# --no_wandb