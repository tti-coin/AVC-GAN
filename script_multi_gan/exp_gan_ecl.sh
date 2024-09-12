# ECl
# python run_gan.py \
# --is_gan_training 1 \
# --is_gan_evaluating 0 \
# --root_path ./data/iTransformer_datasets/electricity/ \
# --data_path electricity.csv \
# --model_id exp_multi_lr0001_bsz32 \
# --model iTransformer \
# --d_model 128 \
# --data custom \
# --features M \
# --seq_len 432 \
# --pred_len 288 \
# --e_layers 2 \
# --enc_in 321 \
# --dec_in 321 \
# --c_out 321 \
# --des ecl \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --gan_model_id exp_gan_multi_hidden_gp \
# --gan_batch_size 1024 \
# --gen_lr 0.0001 \
# --disc_lr 0.0001 \
# --noise_dim 128 \
# --d_update 5 \
# --no_wandb

python run_gan.py \
--is_gan_training 1 \
--is_gan_evaluating 0 \
--root_path ./data/iTransformer_datasets/electricity/ \
--data_path electricity.csv \
--model_id exp_multi_lr0001_bsz32 \
--model iTransformer \
--d_model 128 \
--data custom \
--features M \
--seq_len 432 \
--pred_len 288 \
--e_layers 2 \
--enc_in 321 \
--dec_in 321 \
--c_out 321 \
--des ecl \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001 \
--gan_model_id exp_gan_multi_hidden_gp \
--gan_batch_size 1024 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--noise_dim 128 \
--d_update 5 \
--sample_size 4096 \
--load_iter 19999