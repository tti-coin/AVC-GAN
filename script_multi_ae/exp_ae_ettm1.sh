# ETTm1
# CUDA_VISIMBLE_DEVICES=1

python run_ae.py \
--is_ae_training 1 \
--root_path ./data/iTransformer_datasets/ETT-small/ \
--data_path ETTm1.csv \
--model_id masked_ae \
--model iTransformer \
--d_model 128 \
--data ETTm1 \
--features M \
--seq_len 192 \
--pred_len 192 \
--e_layers 2 \
--enc_in 7 \
--dec_in 7 \
--c_out 7 \
--des ettm1 \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001 \
--no_wandb
# --wandb_notes "Experiments to non-conditioal AE predict 192 steps"

# python run_ae.py \
# --is_ae_training 0 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm1.csv \
# --model_id masked_ae \
# --model iTransformer \
# --d_model 128 \
# --data ETTm1 \
# --features M \
# --seq_len 192 \
# --pred_len 192 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm1 \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --no_wandb
# --wandb_notes "Experiments to non-conditioal AE predict 192 steps"

# python run_ae.py \
# --is_ae_training 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm1.csv \
# --model_id ae \
# --model iTransformer \
# --d_model 128 \
# --data ETTm1 \
# --features M \
# --seq_len 288 \
# --pred_len 288 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm1 \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --wandb_notes "Experiments to non-conditioal AE predict 288 steps"

# python run_ae.py \
# --is_ae_training 0 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm1.csv \
# --model_id ae \
# --model iTransformer \
# --d_model 128 \
# --data ETTm1 \
# --features M \
# --seq_len 288 \
# --pred_len 288 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm1 \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --wandb_notes "Experiments to non-conditioal AE predict 288 steps"

# python run_ae.py \
# --is_ae_training 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm1.csv \
# --model_id ae \
# --model iTransformer \
# --d_model 128 \
# --data ETTm1 \
# --features M \
# --seq_len 432 \
# --pred_len 432 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm1 \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --wandb_notes "Experiments to non-conditioal AE predict 432 steps"

# python run_ae.py \
# --is_ae_training 0 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm1.csv \
# --model_id ae \
# --model iTransformer \
# --d_model 128 \
# --data ETTm1 \
# --features M \
# --seq_len 432 \
# --pred_len 432 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm1 \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --wandb_notes "Experiments to non-conditioal AE predict 432 steps"


# python run_ae.py \
# --is_ae_training 1 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm1.csv \
# --model_id ae \
# --model iTransformer \
# --d_model 128 \
# --data ETTm1 \
# --features M \
# --seq_len 816 \
# --pred_len 816 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm1 \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --wandb_notes "Experiments to non-conditioal AE predict 816 steps"

# python run_ae.py \
# --is_ae_training 0 \
# --root_path ./data/iTransformer_datasets/ETT-small/ \
# --data_path ETTm1.csv \
# --model_id ae \
# --model iTransformer \
# --d_model 128 \
# --data ETTm1 \
# --features M \
# --seq_len 816 \
# --pred_len 816 \
# --e_layers 2 \
# --enc_in 7 \
# --dec_in 7 \
# --c_out 7 \
# --des ettm1 \
# --ae_batch_size 32 \
# --train_epochs 100 \
# --patience 10 \
# --learning_rate 0.001 \
# --wandb_notes "Experiments to non-conditioal AE predict 816 steps"