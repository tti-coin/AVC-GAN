# lr, mask ration チューニング

# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id masked_ae_time_emb \
# --ae_model iTransformer \
# --d_model 128 \
# --mask_ratio 0.7 \
# --d_ff 128 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --learning_rate 0.001 \
# --use_norm 1 \
# --no_wandb


# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id masked_ae_time_emb \
# --ae_model iTransformer \
# --d_model 128 \
# --mask_ratio 0.7 \
# --d_ff 128 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --learning_rate 0.05 \
# --no_wandb



# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id masked_ae_time_emb \
# --ae_model iTransformer \
# --d_model 128 \
# --mask_ratio 0.7 \
# --d_ff 128 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --learning_rate 0.01 \
# --no_wandb


# mask ratio 0.5
# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id masked_ae \
# --ae_model iTransformer \
# --d_model 128 \
# --mask_ratio 0.5 \
# --d_ff 128 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --learning_rate 0.001 \
# --use_norm 1 \
# --no_wandb


# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id masked_ae_time_emb \
# --ae_model iTransformer \
# --d_model 128 \
# --mask_ratio 0.5 \
# --d_ff 128 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --learning_rate 0.05 \
# --no_wandb



# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id masked_ae_time_emb \
# --ae_model iTransformer \
# --d_model 128 \
# --mask_ratio 0.5 \
# --d_ff 128 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --learning_rate 0.01 \
# --no_wandb


# mask ratio 0.2
# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id masked_ae_time_emb \
# --ae_model iTransformer \
# --d_model 128 \
# --mask_ratio 0.2 \
# --d_ff 128 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --learning_rate 0.001 \
# --use_norm 1 \
# --no_wandb


# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id masked_ae_time_emb \
# --ae_model iTransformer \
# --d_model 128 \
# --mask_ratio 0.2 \
# --d_ff 128 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --learning_rate 0.05 \
# --no_wandb



# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id masked_ae_time_emb \
# --ae_model iTransformer \
# --d_model 128 \
# --mask_ratio 0.2 \
# --d_ff 128 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --learning_rate 0.01 \
# --no_wandb


# debug
# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id non-masked_ae \
# --ae_model iTransformer \
# --d_model 256 \
# --d_ff 512 \
# --vari_masked_ratio 0.3 \
# --mask_ratio 0.0 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --patience 3 \
# --learning_rate 0.001 \
# --use_norm 0 \
# --no_wandb


# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id new_masked_ae_wo_maskinfo \
# --ae_model iTransformer \
# --d_model 128 \
# --d_ff 256 \
# --vari_masked_ratio 0.3 \
# --mask_ratio 0.7 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 20 \
# --learning_rate 0.001 \
# --use_norm 0 \
# --no_wandb

# set dec AE
python run.py \
--training_ae 1 \
--root_path ./data/iTransformer_datasets/ETT-small/ \
--data_path ETTm2.csv \
--model_id set_dec_ae_wo_mi \
--ae_model iTransformer_SetDec \
--d_model 128 \
--d_ff 256 \
--vari_masked_ratio 0.0 \
--mask_ratio 0.0 \
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
--train_epochs 20 \
--learning_rate 0.001 \
--use_norm 0 \
--no_wandb


# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id new_masked_ae \
# --ae_model iTransformer \
# --d_model 128 \
# --d_ff 256 \
# --vari_masked_ratio 0.3 \
# --mask_ratio 0.7 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --learning_rate 0.001 \
# --use_norm 0 \
# --no_wandb


# python run.py \
# --training_ae 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm2.csv \
# --model_id new_masked_ae \
# --ae_model iTransformer \
# --d_model 128 \
# --d_ff 256 \
# --vari_masked_ratio 0.3 \
# --mask_ratio 0.9 \
# --data ETTm2 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm2 \
# --ae_batch_size 32 \
# --train_epochs 30 \
# --learning_rate 0.001 \
# --use_norm 0 \
# --no_wandb

