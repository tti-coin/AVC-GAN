model_name=iTransformer

CUDA_VISIBLE_DEVICES=0 python run_all.py \
--root_path /workspace/data/iTransformer_datasets/PEMS/ \
--data_path PEMS03.npz \
--ae_model_id PEMS03 \
--ae_model $model_name \
--data PEMS \
--features M \
--seq_len 96 \
--pred_len 96 \
--e_layers 4 \
--enc_in 358 \
--dec_in 358 \
--c_out 358 \
--des pems03 \
--d_model 128 \
--d_ff 512 \
--learning_rate 0.001 \
--use_norm 0 \
--gan_model ConditionalSAGAN \
--gan_model_id CGAN_accum \
--self_attn \
--gan_batch_size 128 \
--accumulation_steps 8 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--d_update 10 \
--gan_iter 40000 \
--load_iter 40000 \
--sample_size 15616



# python -u run.py \
# --is_training 1 \
# --root_path /workspace/data/iTransformer_datasets/PEMS/ \
# --data_path PEMS03.npz \
# --model_id PEMS03_96_48 \
# --model $model_name \
# --data PEMS \
# --features M \
# --seq_len 96 \
# --pred_len 48 \
# --e_layers 4 \
# --enc_in 358 \
# --dec_in 358 \
# --c_out 358 \
# --des pems03 \
# --d_model 512 \
# --d_ff 512 \
# --learning_rate 0.001 \
# --use_norm 0 \
# --gan_model ConditionalSAGAN \
# --gan_model_id CGAN_v2 \
# --self_attn \
# --gan_batch_size 1024 \
# --gen_lr 0.0001 \
# --disc_lr 0.0001 \
# --d_update 10 \
# --gan_iter 40000 \
# --load_iter 40000 \
# --sample_size 8448


# python -u run.py \
# --is_training 1 \
# --root_path /workspace/data/iTransformer_datasets/PEMS/ \
# --data_path PEMS03.npz \
# --model_id PEMS03_96_96 \
# --model $model_name \
# --data PEMS \
# --features M \
# --seq_len 96 \
# --pred_len 96 \
# --e_layers 4 \
# --enc_in 358 \
# --dec_in 358 \
# --c_out 358 \
# --des pems03 \
# --d_model 512 \
# --d_ff 512 \
# --learning_rate 0.001 \
# --use_norm 0 \
# --gan_model ConditionalSAGAN \
# --gan_model_id CGAN_v2 \
# --self_attn \
# --gan_batch_size 1024 \
# --gen_lr 0.0001 \
# --disc_lr 0.0001 \
# --d_update 10 \
# --gan_iter 40000 \
# --load_iter 40000 \
# --sample_size 8448