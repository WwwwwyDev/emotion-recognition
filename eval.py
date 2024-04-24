from dataset import FaceDataset
import torch
from net import MobileNet, MobileNetV2, MobileNetV3Large, MobileNetV3Small, Vgg19, ResNet50
import numpy as np
import csv

if __name__ == '__main__':
    data = [
        ['ID', 'Emotion'],
    ]
    test_dataset = FaceDataset(csv_path="./expertclass2/test_data.csv", is_test=True)
    net = ResNet50()
    checkpoint = torch.load(f"./{net.name}.pt", map_location="cpu")
    net.load_state_dict(checkpoint)
    net.eval()
    for i, x in enumerate(test_dataset):
        y = net(x)
        pred = np.argmax(y.data.numpy(), axis=1)[0]
        data.append([i, pred])
        if i and i % 100 == 0:
            print(i)
    with open(f'submit{net.name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

