# learning_rate 0.0001
# n_heads 4
python /workspace/run_all.py \
--root_path /workspace/data/iTransformer_datasets/ETT-small/ \
--data_path ETTh2.csv \
--ae_model_id turning_ae \
--ae_model iTransformer \
--d_model 64 \
--d_ff 128 \
--n_heads 4 \
--data ETTh2 \
--features M \
--seq_len 432 \
--pred_len 432 \
--e_layers 2 \
--enc_in 7 \
--dec_in 7 \
--c_out 7 \
--des etth2 \
--ae_batch_size 32 \
--learning_rate 0.0001 \
--use_norm 0 \
--gan_model ConditionalSAGAN \
--gan_model_id CGAN_v2 \
--self_attn \
--gan_batch_size 1024 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--d_update 10 \
--gan_iter 40000 \
--load_iter 40000 \
--sample_size 8208


python /workspace/run_all.py \
--root_path /workspace/data/iTransformer_datasets/ETT-small/ \
--data_path ETTh2.csv \
--ae_model_id turning_ae \
--ae_model iTransformer \
--d_model 128 \
--d_ff 256 \
--n_heads 4 \
--data ETTh2 \
--features M \
--seq_len 432 \
--pred_len 432 \
--e_layers 2 \
--enc_in 7 \
--dec_in 7 \
--c_out 7 \
--des etth2 \
--ae_batch_size 32 \
--learning_rate 0.0001 \
--use_norm 0 \
--gan_model ConditionalSAGAN \
--gan_model_id CGAN_v2 \
--self_attn \
--gan_batch_size 1024 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--d_update 10 \
--gan_iter 40000 \
--load_iter 40000 \
--sample_size 8208


# n_heads 8
python /workspace/run_all.py \
--root_path /workspace/data/iTransformer_datasets/ETT-small/ \
--data_path ETTh2.csv \
--ae_model_id turning_ae \
--ae_model iTransformer \
--d_model 64 \
--d_ff 128 \
--n_heads 8 \
--data ETTh2 \
--features M \
--seq_len 432 \
--pred_len 432 \
--e_layers 2 \
--enc_in 7 \
--dec_in 7 \
--c_out 7 \
--des etth2 \
--ae_batch_size 32 \
--learning_rate 0.0001 \
--use_norm 0 \
--gan_model ConditionalSAGAN \
--gan_model_id CGAN_v2 \
--self_attn \
--gan_batch_size 1024 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--d_update 10 \
--gan_iter 40000 \
--load_iter 40000 \
--sample_size 8208


python /workspace/run_all.py \
--root_path /workspace/data/iTransformer_datasets/ETT-small/ \
--data_path ETTh2.csv \
--ae_model_id turning_ae \
--ae_model iTransformer \
--d_model 128 \
--d_ff 256 \
--n_heads 8 \
--data ETTh2 \
--features M \
--seq_len 432 \
--pred_len 432 \
--e_layers 2 \
--enc_in 7 \
--dec_in 7 \
--c_out 7 \
--des etth2 \
--ae_batch_size 32 \
--learning_rate 0.0001 \
--use_norm 0 \
--gan_model ConditionalSAGAN \
--gan_model_id CGAN_v2 \
--self_attn \
--gan_batch_size 1024 \
--gen_lr 0.0001 \
--disc_lr 0.0001 \
--d_update 10 \
--gan_iter 40000 \
--load_iter 40000 \
--sample_size 8208