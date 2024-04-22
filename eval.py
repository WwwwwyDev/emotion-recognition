from dataset import FaceDataset
import torch
from net import MobileNet
if __name__ == '__main__':
    test_dataset = FaceDataset(csv_path="./expertclass2/train_data.csv", is_test=True)
    net = torch.load("./emotion_model.pt")
    net.eval()
    for x in test_dataset[:2]:
        y = net(x)
        print(y)
    # pred = np.argmax(pred.data.numpy(), axis=1)
    # labels = labels.data.numpy()