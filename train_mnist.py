#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torchvision.datasets as Dataset
import torchvision.transforms as Transform
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import argparse
import struct
import numpy as np

# 从命令行解析相关参数，确定是用来做训练还是做转码
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--method', default = 'transfer', help = 'method: transfer for transfer zoo/lenet.pth to zoo/layer_data.bin; train for train net and save to zoo/lenet.pth')
args = parser.parse_args()

# 网络结构的定义，同时将网络结构写入文件中方便读取相关的参数
# 参考了LeNet的设计，但是由于是在mnist数据集上进行测试，所以只有两层卷积+两层全连接
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        self.conv2 = nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 2)
        self.fc3 = nn.Linear(in_features = 4*4*50, out_features = 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features = 128, out_features = 10)
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x
net = LeNet()
print('网络结构为')
print(net)

# 这部分将结果写到一个模型定义文件中去，目前准备设计的结构只支持单路径计算，
# 对卷积不支持pad，不支持stride，非线性层只支持ReLU，Pooling不支持stride，要求长宽均能整除kernel_size
# 只支持方形的卷积核和降采样核，Pooling支持MAX--0和AVE--1
with open('layer_params.txt', 'w') as f:
    # conv写入conv in_channels out_channels kernel_size 0/1(bias)
    # relu写入relu
    # pooling写入pooling kernel_size 0/1(max or ave)
    # fc写入fc in_features out_features 0/1(bias)
    f.write(f'conv {net.conv1.in_channels} {net.conv1.out_channels} {net.conv1.kernel_size[0]} {int(isinstance(net.conv1.bias, nn.Parameter))}\n')
    f.write('relu\n')
    f.write(f'pooling {net.pool1.kernel_size} 0\n')
    f.write(f'conv {net.conv2.in_channels} {net.conv2.out_channels} {net.conv2.kernel_size[0]} {int(isinstance(net.conv2.bias, nn.Parameter))}\n')
    f.write('relu\n')
    f.write(f'pooling {net.pool2.kernel_size} 0\n')
    f.write(f'fc {net.fc3.in_features} {net.fc3.out_features} {int(isinstance(net.fc3.bias, nn.Parameter))}\n')
    f.write('relu\n')
    f.write(f'fc {net.fc4.in_features} {net.fc4.out_features} {int(isinstance(net.fc4.bias, nn.Parameter))}\n')

# 数据的输入
TRAIN_BATCH_SIZE = 128
TRAIN_NUM_WORKERS = 4
TEST_BATCH_SIZE = 100
TEST_NUM_WORKERS = 4
train_dataset = Dataset.MNIST(root = './mnist',
                              train = True,
                              download = True,
                              transform = Transform.ToTensor(),
                              )
train_loader = DataLoader(dataset = train_dataset,
                          batch_size = TRAIN_BATCH_SIZE,
                          shuffle = True,
                          num_workers = TRAIN_NUM_WORKERS,
                          drop_last = True,
                          )
test_dataset = Dataset.MNIST(root = './mnist',
                             train = False,
                             download = True,
                             transform = Transform.ToTensor(),
                             )
test_loader = DataLoader(dataset = test_dataset,
                         batch_size = TEST_BATCH_SIZE,
                         shuffle = False,
                         num_workers = TEST_NUM_WORKERS,
                         drop_last = False,
                         )
print(f'训练数据集图片数为 {len(train_dataset)}, iteration数为{len(train_loader)}')
print(f'测试数据集图片数为 {len(test_dataset)}, iteration数为{len(test_loader)}')

# 网络的训练策略
BASE_LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
MILESTONES = [10]
GAMMA = 0.1
EPOCHS = 20

TRAIN_PARAMETER = \
'''# 训练的相关参数为
## loss
CrossEntropyLoss
## optimizer
SGD: base_lr %f momentum %f weight_decay %f
## lr_policy
MultiStepLR: milestones [%s] gamma %f epochs %d'''\
%(
BASE_LR,
MOMENTUM,
WEIGHT_DECAY,
', '.join(str(e) for e in MILESTONES),
GAMMA,
EPOCHS,
)
print(TRAIN_PARAMETER)
# 定义测试网络
def eval_net(epoch):
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            test_total += labels.size(0)
            # predicted
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
    print('%s After epoch %d, accuracy is %2.4f' % (time.asctime(time.localtime(time.time())), epoch, test_correct / test_total))
# 定义训练网络
if args.method == 'train':
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = BASE_LR, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = MILESTONES, gamma = GAMMA)
    # initial test
    eval_net(0)
    # epochs
    for epoch in range(EPOCHS):
        # train
        net.train()
        scheduler.step()
        for i, (images, labels) in enumerate(train_loader):
            net.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'epoch {epoch+1:3d}, {i:3d}|{len(train_loader):3d}, loss: {loss.item():2.4f}', end = '\r')
        eval_net(epoch + 1)
        torch.save(net.state_dict(), f'zoo/lenet.pth')
#  这部分用来实现将params转换为c语言较好读取的格式
if args.method == 'transfer':
    net.load_state_dict(torch.load('zoo/lenet.pth'))
    # 测试精度
    eval_net(0)
    # 将参数写入到文件中去
    with open('zoo/layer_data.bin', 'wb') as f:
        for param in net.parameters():
            print(param.shape)
            # 先转成np矩阵，再按照C格式reshape，再转成list
            tmp = param.detach().numpy()
            total_length = tmp.size
            tmp = np.reshape(tmp, (total_length), order='C')
            tmp = tmp.tolist()
            # 利用struct方式写入文件，*将元组解析
            f.write(struct.pack(f'{total_length}f', *tuple(tmp)))
