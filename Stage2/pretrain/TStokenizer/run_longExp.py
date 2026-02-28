import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

    # basic config
    parser.add_argument('--model', type=str, default='SVQ',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/solar_stations/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Solar station site 8 (Nominal capacity-30MW).csv', help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--scale', type=bool, default=True, help='Whether to scale the input data (default: True)')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    # Sparse-VQ
    parser.add_argument('--codebook_size', type=int, default=128, help='codebook_size in sparse vector quantized')

    parser.add_argument('--attn_dropout', type=float, default=0.0, help='attn dropout')
    parser.add_argument('--patch_len', type=int, default=4, help='patch length')

    # Formers 
    parser.add_argument('--enc_in', type=int, default=4, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='Huber', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    parser.add_argument('--quantizer_name', default='MGVQ', help='the name of models.',
                        choices=['FQGAN','MGVQ', 'VectorQuantizerSinkhorn', 'wasserstein_quantizer', 'Vanilla_Quantizer', 'ema_quantizer', 'online_quantizer','VectorQuantize'])
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=0.9)

    args = parser.parse_args()

    # random seed
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    Exp = Exp_Main

    setting = 'data_path_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_lr{}_batch{}_loss{}_codebook_size{}_delta{}_sparsity{}_dropout{}'.format(
        args.data_path,
        args.seq_len,
        args.patch_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.learning_rate,
        args.batch_size,
        args.loss,
        args.codebook_size, args.delta, args.sparsity, args.dropout)
    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    torch.cuda.empty_cache()
