# Weather
CUDA_VISIMBLE_DEVICES=0

python run_ae.py \
--is_ae_training 1 \
--root_path ./data/iTransformer_datasets/weather/ \
--data_path weather.csv \
--model_id ae \
--model iTransformer \
--d_model 128 \
--data custom \
--features M \
--seq_len 192 \
--pred_len 192 \
--e_layers 2 \
--enc_in 21 \
--dec_in 21 \
--c_out 21 \
--des weather \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001 \
--wandb_notes "Experiments to non-conditioal AE predict 192 steps"

python run_ae.py \
--is_ae_training 0 \
--root_path ./data/iTransformer_datasets/weather/ \
--data_path weather.csv \
--model_id ae \
--model iTransformer \
--d_model 128 \
--data custom \
--features M \
--seq_len 192 \
--pred_len 192 \
--e_layers 2 \
--enc_in 21 \
--dec_in 21 \
--c_out 21 \
--des weather \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001 \
--wandb_notes "Experiments to non-conditioal AE predict 192 steps"

python run_ae.py \
--is_ae_training 1 \
--root_path ./data/iTransformer_datasets/weather/ \
--data_path weather.csv \
--model_id ae \
--model iTransformer \
--d_model 128 \
--data custom \
--features M \
--seq_len 288 \
--pred_len 288 \
--e_layers 2 \
--enc_in 21 \
--dec_in 21 \
--c_out 21 \
--des weather \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001 \
--wandb_notes "Experiments to non-conditioal AE predict 288 steps"

python run_ae.py \
--is_ae_training 0 \
--root_path ./data/iTransformer_datasets/weather/ \
--data_path weather.csv \
--model_id ae \
--model iTransformer \
--d_model 128 \
--data custom \
--features M \
--seq_len 288 \
--pred_len 288 \
--e_layers 2 \
--enc_in 21 \
--dec_in 21 \
--c_out 21 \
--des weather \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001 \
--wandb_notes "Experiments to non-conditioal AE predict 288 steps"

python run_ae.py \
--is_ae_training 1 \
--root_path ./data/iTransformer_datasets/weather/ \
--data_path weather.csv \
--model_id ae \
--model iTransformer \
--d_model 128 \
--data custom \
--features M \
--seq_len 432 \
--pred_len 432 \
--e_layers 2 \
--enc_in 21 \
--dec_in 21 \
--c_out 21 \
--des weather \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001 \
--wandb_notes "Experiments to non-conditioal AE predict 432 steps"

python run_ae.py \
--is_ae_training 0 \
--root_path ./data/iTransformer_datasets/weather/ \
--data_path weather.csv \
--model_id ae \
--model iTransformer \
--d_model 128 \
--data custom \
--features M \
--seq_len 432 \
--pred_len 432 \
--e_layers 2 \
--enc_in 21 \
--dec_in 21 \
--c_out 21 \
--des weather \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001 \
--wandb_notes "Experiments to non-conditioal AE predict 432 steps"


python run_ae.py \
--is_ae_training 1 \
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
--wandb_notes "Experiments to non-conditioal AE predict 816 steps"

python run_ae.py \
--is_ae_training 0 \
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
--wandb_notes "Experiments to non-conditioal AE predict 816 steps"