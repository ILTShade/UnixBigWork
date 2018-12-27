#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as Dataset
import torchvision.transforms as Transform
from torch.utils.data import DataLoader

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
    def forward(x):
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
