from dataset import FaceDataset
import torch
from net import MobileNet
import numpy as np
import csv

if __name__ == '__main__':
    data = [
        ['ID', 'Emotion'],
    ]
    test_dataset = FaceDataset(csv_path="./expertclass2/train_data.csv", is_test=True)
    net = MobileNet()
    checkpoint = torch.load("./emotion_model.pt", map_location="cpu")
    net.load_state_dict(checkpoint)
    net.eval()
    for i, x in enumerate(test_dataset):
        y = net(x)
        pred = np.argmax(y.data.numpy(), axis=1)[0]
        data.append([i, pred])
    filename = 'submit.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

