# python /workspace/run_all.py \
# --root_path /workspace/data/timegan_datasets \
# --data_path energy_data.csv \
# --ae_model_id short_ae \
# --ae_model iTransformer \
# --d_model 128 \
# --d_ff 512 \
# --n_heads 8 \
# --data energy \
# --features M \
# --seq_len 48 \
# --pred_len 48 \
# --e_layers 2 \
# --enc_in 28 \
# --dec_in 28 \
# --c_out 28 \
# --des energy \
# --ae_batch_size 32 \
# --learning_rate 0.0001 \
# --use_norm 0 \
# --gan_model ConditionalSAGAN \
# --gan_model_id CGAN_short \
# --self_attn \
# --gan_batch_size 1024 \
# --accumulation_steps 1 \
# --gen_lr 0.0001 \
# --disc_lr 0.0001 \
# --d_update 10 \
# --gan_iter 40000 \
# --load_iter 40000 \
# --sample_size 13766

python /workspace/run_all.py \
--root_path /workspace/data/timegan_datasets \
--data_path energy_data.csv \
--ae_model_id short_ae \
--ae_model iTransformer \
--d_model 64 \
--d_ff 256 \
--n_heads 8 \
--data energy \
--features M \
--seq_len 48 \
--pred_len 48 \
--e_layers 1 \
--enc_in 28 \
--dec_in 28 \
--c_out 28 \
--des energy \
--ae_batch_size 32 \
--learning_rate 0.0001 \
--use_norm 0 \
--gan_model ConditionalSAGAN \
--gan_model_id CGAN_short \
--self_attn \
--gan_batch_size 1024 \
--accumulation_steps 1 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--d_update 10 \
--gan_iter 40000 \
--load_iter 40000 \
--sample_size 13766