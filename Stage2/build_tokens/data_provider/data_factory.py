# data_provider/data_factory.py
from .data_loader import Dataset_Custom
from torch.utils.data import DataLoader


def data_provider(args):
    data_set = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        size=[args.seq_len, args.pred_len],
        enc_in=args.enc_in,
        target_col=args.target_col,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    print("[Dataset]", len(data_set))
    return data_set, data_loader
