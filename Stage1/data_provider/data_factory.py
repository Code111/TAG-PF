from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    # timeenc = 0 if args.embed != 'timeF' else 1
    
    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        flag=flag,
        size=[args.seq_len],
        data_path=args.data_path,
        scale=args.scale,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
