# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

from dataset import FaceDatasetMLU
from net import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small

import torch_mlu
import os

os.putenv('MLU_VISIBLE_DEVICES', '0')


def train(train_dataset, batch_size, epochs, learning_rate, wt_decay):
    # 载入数据并分割batch
    train_loader = data.DataLoader(train_dataset, batch_size)
    # 构建模型
    model = MobileNet().to("mlu")
    # 损失函数
    loss_function = nn.CrossEntropyLoss().to("mlu")
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_decay)
    loss_record = []
    # 逐轮训练
    for epoch in range(epochs):
        # 记录损失值
        loss_rate = 0
        model.train()  # 模型训练
        for images, labels in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            output = model.forward(images)
            # 误差计算
            loss_rate = loss_function(output, labels)
            # 误差的反向传播
            loss_rate.backward()
            # 更新参数
            optimizer.step()
            loss_record.append(str(loss_rate.item()))

        # 打印每轮的损失
        print('After {} epochs , the loss_rate is : '.format(epoch + 1), loss_rate.item())
        if epoch % 5 == 0:
            text = ','.join(loss_record)
            with open(f"{model.name}_loss.txt", 'w') as file:
                file.write(text)
            torch.save(model.state_dict(), f"{model.name}.pt")
    text = ','.join(loss_record)
    with open(f"{model.name}_loss.txt", 'w') as file:
        file.write(text)
    torch.save(model.state_dict(), f"{model.name}.pt")


if __name__ == '__main__':
    dataset = FaceDatasetMLU(csv_path="./expertclass2/train_data.csv", is_test=False)
    # 超参数可自行指定
    train(train_dataset=dataset, batch_size=128, epochs=20, learning_rate=0.00005, wt_decay=0)
