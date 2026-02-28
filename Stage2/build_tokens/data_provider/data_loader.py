# data_provider/data_loader.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class Dataset_Custom(Dataset):
    """
    Returns:
      seq_x_scaled: [seq_len, C_in]
      seq_y_scaled: [pred_len, 1]
      seq_x_date:   list[str] length=seq_len
      seq_y_date:   list[str] length=pred_len
      seq_x_raw:    [seq_len, C_in]   (未归一化原值)
      seq_y_raw:    [pred_len, 1]     (未归一化原值)
    """

    def __init__(self, root_path, data_path, size, enc_in=4, target_col=-1):
        self.root_path = root_path
        self.data_path = data_path
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.enc_in = enc_in
        self.target_col = target_col
        self.__read_data__()

    def __read_data__(self):
        path = os.path.join(self.root_path, self.data_path)
        df = pd.read_csv(path)

        # 第一列是 date 字符串
        self.dates = df.iloc[:, 0].astype(str).tolist()

        # 后面是数值列
        val_df = df.iloc[:, 1:]
        data = val_df.values.astype(np.float32)  # [N, F]

        if data.shape[1] < self.enc_in:
            raise ValueError(f"CSV has {data.shape[1]} features, but enc_in={self.enc_in}")

        x = data[:, : self.enc_in]        # [N, C_in]
        y = data[:, self.target_col :]    # [N, 1]

        self.data_x_raw = x.astype(np.float32)
        self.data_y_raw = y.astype(np.float32)

        N = len(df)
        num_train = int(N * 0.7)

        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.scaler_x.fit(x[:num_train])
        self.scaler_y.fit(y[:num_train])

        self.data_x = self.scaler_x.transform(x).astype(np.float32)
        self.data_y = self.scaler_y.transform(y).astype(np.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        s = idx
        e = s + self.seq_len
        r0 = e
        r1 = r0 + self.pred_len

        seq_x_scaled = torch.from_numpy(self.data_x[s:e]).float()
        seq_y_scaled = torch.from_numpy(self.data_y[r0:r1]).float()

        seq_x_raw = torch.from_numpy(self.data_x_raw[s:e]).float()
        seq_y_raw = torch.from_numpy(self.data_y_raw[r0:r1]).float()

        seq_x_date = self.dates[s:e]
        seq_y_date = self.dates[r0:r1]

        return seq_x_scaled, seq_y_scaled, seq_x_date, seq_y_date, seq_x_raw, seq_y_raw
