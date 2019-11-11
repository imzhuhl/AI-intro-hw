import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchsummary
from PIL import Image

from utils import Config
from mynet import MyNet

DEVICE = torch.device("cuda:1" if Config.use_cuda else "cpu")


def load_data():
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data = ImageFolder(root=Config.data_path, transform=trans)
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_set, val_set = random_split(data, [train_size, val_size])
    
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=64, shuffle=True)

    return train_loader, val_loader


def train(net, train_loader, val_loader):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-5)

    best_val_loss = None

    for epoch in range(Config.num_epoch):
        epoch_start_time = time.time()

        net.train()  # 训练模式
        train_loss = 0.0
        total = 0
        correct = 0
        for train_x, train_y in train_loader:
            train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
            optimizer.zero_grad()  # 梯度置零
            train_pred = net(train_x)  # 向前传播，求出预测值
            batch_loss = loss(train_pred, train_y)  # 计算 loss func
            batch_loss.backward()  # 反向传播求梯度
            optimizer.step()  # 更新权重 weights 
            train_loss += batch_loss.item()
            correct += np.sum(
                np.argmax(train_pred.cpu().data.numpy(), axis=1) == train_y.cpu().numpy())
            total += len(train_y)
        train_acc = correct * 1.0 / total

        net.eval()
        val_loss = 0.0
        total = 0
        correct = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                pred = net(val_X.to(DEVICE))
                val_loss += loss(pred, val_y.to(DEVICE))
                correct += np.sum(
                    np.argmax(pred.cpu().data.numpy(), axis=1) == val_y.numpy())
                total += len(val_y)
        val_acc = correct * 1.0 / total
          
        print('[{}/{}] {:.2f} sec(s) | loss: {:.2f} | train acc: {:.3f} | val loss: {:.2f} | val acc: {:.3f}'
                .format(epoch+1, 
                        Config.num_epoch, 
                        time.time() - epoch_start_time, 
                        train_loss,
                        train_acc,
                        val_loss,
                        val_acc))
    
        if best_val_loss is None:
            best_val_loss = val_loss
            continue
        
        if best_val_loss > val_loss or epoch % 20 == 0:
            best_val_loss = min(val_loss, best_val_loss)
            file_path = "{}mynet-{}-{:.2f}-{:.2f}.pth".format(
                Config.weight_path, epoch+1, val_loss, val_acc)
            torch.save(net.state_dict(), file_path)
    
    torch.save(net.state_dict(), "./weights/mynet.pth")


def predict(img):
    """
    加载模型和模型预测
    主要步骤:
        1.加载模型(请加载你认为的最佳模型)
        2.图片处理
        3.用加载的模型预测图片的类别
    :param img: PIL.Image 对象
    :return: string, 模型识别图片的类别, 
            共 'cardboard','glass','metal','paper','plastic','trash' 6 个类别
    """
    # 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
    # 如果你的模型是在 results 文件夹下的 dnn.h5 模型，则 model_path = 'results/dnn.h5'
    model_path = "weights/mynet-64-8.16-0.81.pth"
    try:
        # 作业提交时测试用, 请勿删除此部分
        model_path = os.path.realpath(__file__).replace('main.py', model_path)
    except NameError:
        model_path = './' + model_path

    # -------------------------- 实现模型预测部分的代码 ---------------------------
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = trans(img)
    img = torch.unsqueeze(img, dim=0)

    # 加载模型
    net = MyNet().cpu()
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    net.eval()
    with torch.no_grad():
        pred = net(img)
        pred = pred.numpy()
        predict = labels[np.argmax(pred)]
    
    # -------------------------------------------------------------------------

    # 返回图片的类别
    return predict



if __name__ == "__main__":
    # train_loader, val_loader = load_data()
    # net = MyNet().to(DEVICE)
    # torchsummary.summary(net, (3, 512, 384), device="cuda")
    # net.load_state_dict(torch.load("./weights/mynet-4-1.25-0.94.pth"))
    # train(net, train_loader, val_loader)
    img = Image.open("./test.jpg")
    print(predict(img))

    
