import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from model import My_Model
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

writer = SummaryWriter("logs")
transformation = transforms.Compose([
                    transforms.ToTensor(),   # 1.转换成Tensor  2.转换到0-1之间  3.会将channel放到第一维度

])

# 60000
train_ds = datasets.MNIST(
                    'D://pkr_study//codes//02Number_Detection//dataset',          # 数据存放的路径
                    train=True,
                    transform=transformation,
                    download=True             # 如果没有就下载
)

# 10000
test_ds = datasets.MNIST(
                    'D://pkr_study//codes//02Number_Detection//dataset',
                    train=False,              # 注意此处为False
                    transform=transformation,
                    download=True
)

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256)          # 测试集计算量比较少，批次一次可以大点


imgs, labels = next(iter(train_dl))
img = imgs[0]
label = labels[0]
# print(img.shape)
# img = np.squeeze(img)
# plt.imshow(img)
# plt.show()
# print(labels[0])

# 模型、损失函数、优化器
device = "cpu"
model = My_Model().to(device=device)
loss_fn = torch.nn.CrossEntropyLoss().to(device=device)
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

def run_episode(epoch, model, trainloader):
    episode_loss = 0
    episode_acc = 0
    total = 0
    for img, label in trainloader:
        pred = model(img)
        # label = torch.tensor(label, device="cuda:0")
        loss = loss_fn(pred, label)
        pred = torch.argmax(pred, dim=1)
        acc = (label == pred).sum().item()
        total += label.size(0)
        episode_acc += acc
        episode_loss += loss
        # 更新参数
        optim.zero_grad()
        loss.backward()
        optim.step()
    writer.add_scalar("loss", episode_loss/total, epoch)
    writer.add_scalar("acc", episode_acc/total, epoch)
    print(f"epoch:{epoch}--loss:{episode_loss/total}--acc:{episode_acc/total}")

def test(epoch, model, testloader):
    episode_acc = 0
    total = 0
    for img, label in testloader:
        pred = model(img)
        # label = torch.tensor(label, device="cuda:0")
        pred = torch.argmax(pred, dim=1)
        acc = (label == pred).sum().item()
        total += 1
        episode_acc += acc
    writer.add_scalar("test_acc", episode_acc/total, epoch)
    print(f"acc:{episode_acc/total}")

for epoch in range(100):
    run_episode(epoch, model, train_dl)
    test(epoch, model, test_ds)
