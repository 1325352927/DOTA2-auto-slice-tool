import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils
import cv2
import os


# 包含了一个隐含层的全连接神经网络
class NeuralNet(nn.Module):
    # 输入数据的维度，中间层的节点数，输出数据的维度
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


def train(device, train_loader):
    # 输入节点数就为图片的大小：  63*39
    input_size = 100*100
    # 4类输出
    num_classes = 11
    # 建立了一个中间层为 500 的三层神经网络，且将模型转为当前环境支持的类型（CPU 或 GPU）
    model = NeuralNet(input_size, 1000, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    # 此时学习率为 0.001 ，你也可以根据实际情况，自行设置
    learning_rate = 0.001
    # 定义 Adam 优化器用于梯度下降
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 5
    # 数据总长度
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # 因为全连接会把一行数据当做一条数据，因此我们需要将一张图片转换到一行上
            # 原始数据集的大小: [4,1,39,63]
            images = images.reshape(-1, 100*100).to(device)
            labels = labels.to(device)

            # 正向传播以及损失的求取
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            # 下面三句话固定：梯度清空，反向传播，权重更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 10) % 1 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.8f}')
    print("模型训练完成")
    # 保存为可调用对象
    model.eval()
    torch.save(model, "./model/model.pth")
    print("模型保存完成")
    
    return model


def test(model, dataset_path, device):
    imagelist = os.listdir(dataset_path)
    for imgname in imagelist:
        img = cv2.imread(dataset_path + imgname)  # 读取图片
        trans = transforms.ToTensor()
        images = trans(img)
        images = images.reshape(-1, 100 * 100).to(device)
        outputs = model(images)
        num = np.argmax(outputs[0:1].detach().cpu().numpy())
        print(num)

def get_one_num(model, pic_name, device):
    img = cv2.imread(pic_name)  # 读取图片
    print(pic_name)
    trans = transforms.ToTensor()
    images = trans(img)
    images = images.reshape(-1, 100 * 100).to(device)
    outputs = model(images)
    num = np.argmax(outputs[0:1].detach().cpu().numpy())
    return num

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])

    train_dataset = torchvision.datasets.ImageFolder(root='./dataset/train/', transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    # Batch Size定义：一次训练所选取的样本数。 Batch Size的大小影响模型的优化程度和速度。
    # 训练
    model = train(device, train_loader)
    # 加载模型
    model = torch.load("./model/model.pth")
    model.eval()
    print("模型加载完成")
    test(model, './dataset/test/0/', device)

