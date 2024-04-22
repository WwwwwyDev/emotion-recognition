import torch.utils.data as data
import pandas as pd
import numpy as np
import torch


class FaceDataset(data.Dataset):
    # 初始化
    def __init__(self, csv_path, is_test=False):
        super(FaceDataset, self).__init__()
        df = pd.read_csv(csv_path)
        img = list(df["pixels"])
        for index, element in enumerate(img):
            img[index] = list(map(int, element.split(" ")))
        self.img = np.array(img, dtype='uint8')
        if not is_test:
            self.label = np.array(df["emotion"])
        self.is_test = is_test

    # 读取某幅图片，item为索引号
    def __getitem__(self, index):
        x = torch.from_numpy(self.img[index]).reshape(1, 48, 48) / 255.0
        if self.is_test:
            return x
        y = torch.tensor(self.label[index])
        return x, y

    # 获取数据集样本个数
    def __len__(self):
        return self.img.shape[0]


class FaceDatasetMLU(data.Dataset):
    # 初始化
    def __init__(self, csv_path, is_test=False):
        super(FaceDatasetMLU, self).__init__()
        df = pd.read_csv(csv_path)
        img = list(df["pixels"])
        for index, element in enumerate(img):
            img[index] = list(map(int, element.split(" ")))
        self.img = np.array(img, dtype='uint8')
        if not is_test:
            self.label = np.array(df["emotion"])
        self.is_test = is_test

    # 读取某幅图片，item为索引号
    def __getitem__(self, index):
        x = torch.from_numpy(self.img[index]).reshape(1, 48, 48) / 255.0
        if self.is_test:
            return x.to("mlu")
        y = torch.tensor(self.label[index])
        return x.to("mlu"), y.to("mlu")

    # 获取数据集样本个数
    def __len__(self):
        return self.img.shape[0]

