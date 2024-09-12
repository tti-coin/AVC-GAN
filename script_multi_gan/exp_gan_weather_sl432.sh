# python run_gan.py \
# --is_gan_training 1 \
# --is_gan_evaluating 0 \
# --root_path ./data/iTransformer_datasets/weather/ \
# --data_path weather.csv \
# --model_id ae \
# --model iTransformer \
# --d_model 128 \
# --data custom \
# --features M \
# --seq_len 432 \
# --pred_len 432 \
# --e_layers 2 \
# --enc_in 21 \
# --dec_in 21 \
# --c_out 21 \
# --des weather \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --gan_model_id cgan \
# --use_hidden \
# --gan_batch_size 1024 \
# --gen_lr 0.0001 \
# --disc_lr 0.0001 \
# --noise_dim 128 \
# --d_update 10 \
# --gan_iter 20000 \
# --wandb_notes "Experiments to generate 432 steps with non-conditional AutoEncoder"


# Evaluation
# python run_gan.py \
# --is_gan_training 0 \
# --is_gan_evaluating 1 \
# --root_path ./data/iTransformer_datasets/weather/ \
# --data_path weather.csv \
# --model_id ae \
# --model iTransformer \
# --d_model 128 \
# --data custom \
# --features M \
# --seq_len 432 \
# --pred_len 432 \
# --e_layers 2 \
# --enc_in 21 \
# --dec_in 21 \
# --c_out 21 \
# --des weather \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --gan_model_id cgan \
# --use_hidden \
# --gan_batch_size 1024 \
# --gen_lr 0.0001 \
# --disc_lr 0.0001 \
# --noise_dim 128 \
# --d_update 10 \
# --gan_iter 20000 \
# --sample_size 4096 \
# --load_iter 20000 \
# --wandb_notes "Experiments to generate 432 steps with non-conditional AutoEncoder" \
# --no_wandb


# python run_gan.py \
# --is_gan_training 1 \
# --is_gan_evaluating 0 \
# --root_path ./data/iTransformer_datasets/weather/ \
# --data_path weather.csv \
# --model_id ae \
# --model iTransformer \
# --d_model 128 \
# --data custom \
# --features M \
# --seq_len 816 \
# --pred_len 816 \
# --e_layers 2 \
# --enc_in 21 \
# --dec_in 21 \
# --c_out 21 \
# --des weather \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --gan_model_id cgan \
# --use_hidden \
# --gan_batch_size 1024 \
# --gen_lr 0.0001 \
# --disc_lr 0.0001 \
# --noise_dim 128 \
# --d_update 10 \
# --gan_iter 20000 \
# --wandb_notes "Experiments to generate 816 steps with non-conditional AutoEncoder"


# Evaluation
python run_cgan.py \
--is_gan_training 0 \
--is_gan_evaluating 1 \
--root_path ./data/iTransformer_datasets/weather/ \
--data_path weather.csv \
--model_id ae \
--model iTransformer \
--d_model 128 \
--data custom \
--features M \
--seq_len 816 \
--pred_len 816 \
--e_layers 2 \
--enc_in 21 \
--dec_in 21 \
--c_out 21 \
--des weather \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001 \
--gan_model_id cgan \
--use_hidden \
--gan_batch_size 1024 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--noise_dim 128 \
--d_update 10 \
--gan_iter 20000 \
--sample_size 4096 \
--load_iter 20000 \
--no_wandb


python run_cgan.py \
--is_gan_training 0 \
--is_gan_evaluating 0 \
--is_synthesizig 1 \
--root_path ./data/iTransformer_datasets/weather/ \
--data_path weather.csv \
--model_id ae \
--model iTransformer \
--d_model 128 \
--data custom \
--features M \
--seq_len 816 \
--pred_len 816 \
--e_layers 2 \
--enc_in 21 \
--dec_in 21 \
--c_out 21 \
--des weather \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001 \
--gan_model_id cgan \
--use_hidden \
--gan_batch_size 1024 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--noise_dim 128 \
--d_update 10 \
--gan_iter 20000 \
--no_wandb \
--load_iter 20000 \
--sample_size 36071