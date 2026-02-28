python -u run_longExp.py --codebook_size 64 --n_heads 16 --e_layers 2 --d_layers 1 --loss Huber --learning_rate 0.001 --delta 0.9 --sparsity 0.5 --d_model 512 --d_ff 1024 --data_path 'Solar station site 1 (Nominal capacity-50MW).csv' --dropout 0.1

python -u run_longExp.py --codebook_size 128 --n_heads 16 --e_layers 2 --d_layers 1 --loss Huber --learning_rate 0.001 --delta 0.9 --sparsity 0.5 --d_model 512 --d_ff 1024 --data_path 'Solar station site 1 (Nominal capacity-50MW).csv' --dropout 0.1

python -u run_longExp.py --codebook_size 256 --n_heads 16 --e_layers 2 --d_layers 1 --loss Huber --learning_rate 0.001 --delta 0.9 --sparsity 0.5 --d_model 512 --d_ff 1024 --data_path 'Solar station site 1 (Nominal capacity-50MW).csv' --dropout 0.1

python -u run_longExp.py --codebook_size 512 --n_heads 16 --e_layers 2 --d_layers 1 --loss Huber --learning_rate 0.001 --delta 0.9 --sparsity 0.5 --d_model 512 --d_ff 1024 --data_path 'Solar station site 1 (Nominal capacity-50MW).csv' --dropout 0.1

python -u run_longExp.py --codebook_size 1024 --n_heads 16 --e_layers 2 --d_layers 1 --loss Huber --learning_rate 0.001 --delta 0.9 --sparsity 0.5 --d_model 512 --d_ff 1024 --data_path 'Solar station site 1 (Nominal capacity-50MW).csv' --dropout 0.1

python -u run_longExp.py --codebook_size 2048 --n_heads 16 --e_layers 2 --d_layers 1 --loss Huber --learning_rate 0.001 --delta 0.9 --sparsity 0.5 --d_model 512 --d_ff 1024 --data_path 'Solar station site 1 (Nominal capacity-50MW).csv' --dropout 0.1

python -u run_longExp.py --codebook_size 4096 --n_heads 16 --e_layers 2 --d_layers 1 --loss Huber --learning_rate 0.001 --delta 0.9 --sparsity 0.5 --d_model 512 --d_ff 1024 --data_path 'Solar station site 1 (Nominal capacity-50MW).csv' --dropout 0.1
