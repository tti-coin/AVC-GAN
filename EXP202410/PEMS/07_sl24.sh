model_name=iTransformer

CUDA_VISIBLE_DEVICES=1 python run_all.py \
--training_gan 1 \
--root_path /workspace/data/iTransformer_datasets/PEMS/ \
--data_path PEMS07.npz \
--ae_model_id PEMS07 \
--ae_model $model_name \
--data PEMS \
--features M \
--seq_len 24 \
--pred_len 24 \
--e_layers 4 \
--enc_in 883 \
--dec_in 883 \
--c_out 883 \
--des pems07 \
--d_model 128 \
--d_ff 512 \
--learning_rate 0.001 \
--use_norm 0 \
--gan_model ConditionalSAGAN \
--gan_model_id CGAN_auum \
--self_attn \
--gan_batch_size 64 \
--accumulation_steps 16 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--d_update 10 \
--gan_iter 40000 \
--load_iter 40000 \
--sample_size 16896

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS07.npz \
#   --model_id PEMS07_96_24 \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 24 \
#   --e_layers 2 \
#   --enc_in 883 \
#   --dec_in 883 \
#   --c_out 883 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --learning_rate 0.001 \
#   --itr 1 \
#   --use_norm 0

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS07.npz \
#   --model_id PEMS07_96_48 \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 48 \
#   --e_layers 4 \
#   --enc_in 883 \
#   --dec_in 883 \
#   --c_out 883 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 16\
#   --learning_rate 0.001 \
#   --itr 1 \
#   --use_norm 0

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/PEMS/ \
#   --data_path PEMS07.npz \
#   --model_id PEMS07_96_96 \
#   --model $model_name \
#   --data PEMS \
#   --features M \
#   --seq_len 96 \
#   --pred_len 96 \
#   --e_layers 4 \
#   --enc_in 883 \
#   --dec_in 883 \
#   --c_out 883 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --batch_size 16\
#   --learning_rate 0.001 \
#   --itr 1 \
#   --use_norm 0
