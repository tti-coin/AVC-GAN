# Traffic
python run_ae.py \
--is_ae_training 1 \
--root_path ./data/informer/traffic/ \
--data_path traffic.csv \
--model_id exp_multi_lr0001_bsz32 \
--model iTransformer \
--d_model 128 \
--data custom \
--features M \
--seq_len 96 \
--pred_len 256 \
--e_layers 2 \
--enc_in 862 \
--dec_in 862 \
--c_out 862 \
--des traffic \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001


python run_ae.py \
--is_ae_training 0 \
--root_path ./data/informer/traffic/ \
--data_path traffic.csv \
--model_id exp_multi_lr0001_bsz32 \
--model iTransformer \
--d_model 128 \
--data custom \
--features M \
--seq_len 96 \
--pred_len 256 \
--e_layers 2 \
--enc_in 862 \
--dec_in 862 \
--c_out 862 \
--des traffic \
--ae_batch_size 32 \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.001
