# data_provider/data_factory.py 

import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from .data_loader import TokensPTDataset


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def data_provider(args, collate_fn):
    ds = TokensPTDataset(args.root_path, args.data_path)
    n = len(ds)

    n_train = int(args.train_ratio * n)
    n_val = int(args.val_ratio * n)

    train_ds = Subset(ds, range(0, n_train))
    val_ds = Subset(ds, range(n_train, n_train + n_val))
    test_ds = Subset(ds, range(n_train + n_val, n))

    # samplers
    if _is_distributed():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=True)
        test_sampler = DistributedSampler(test_ds, shuffle=False, drop_last=True)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True

    # dataloader configs
    num_workers = int(getattr(args, "num_workers", 0))
    pin_memory = bool(getattr(args, "pin_memory", True))
    persistent_workers = bool(getattr(args, "persistent_workers", num_workers > 0))
    persistent_workers = persistent_workers and (num_workers > 0)

    prefetch_factor = getattr(args, "prefetch_factor", 2)
    if num_workers <= 0:
        prefetch_factor = None

    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if prefetch_factor is not None:
        common_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.train_batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        **common_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        **common_kwargs,
    )

    # generate eval loader: true distributed split WITHOUT duplicates
    if _is_distributed():
        ws = dist.get_world_size()
        rk = dist.get_rank()
        idxs = list(range(rk, len(test_ds), ws))  # rk, rk+ws, rk+2ws...
        raw_test_ds = Subset(test_ds, idxs)
    else:
        raw_test_ds = test_ds

    raw_test_loader = DataLoader(
        raw_test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=True,
        **common_kwargs,
    )

    return ds.meta, train_loader, val_loader, test_loader, raw_test_loader
