"""
2020.7.14
网络输入采用200Hz降采样+滤波后的脑电数据，
通道数：62
"""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import os
import scipy.io

channels = 62
timepoints = 200


# 定义网络模型
class EEGNet(nn.Module):
    def __init__(self):  # 初始化网络架构
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        # 二维卷积，输入通道数1，输出通道数16，卷积核大小1*64，零填充
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        # 数据归一化处理，affine参数设为True表示weight和bias将被使用

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        # 用0填充输入张量边界
        self.conv2 = nn.Conv2d(1, 4, (2, 32))  # 二维卷积
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        # 对于输入信号的输入通道，提供2维最大池化操作

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # 全连接层
        # 此维度将取决于数据中每个样本的时间戳数。
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4 * 2 * 7, 1)  # 对输入数据进行一个线性变换


    def forward(self, x):  # 前向传播，由输入获取输出的一个过程
        # 输入x是一个张量
        # layer 1
        x = nn.ZeroPad2d((50, 49, 0, 0))(x)
        x = nn.Conv2d(1, 8, (1, 100), padding=0)(x)
        x = nn.BatchNorm2d(8, False)(x)  # 数据做一个归一化

        # Layer 2
        x = nn.Conv2d(8, 16, (62, 1))(x)
        x = nn.BatchNorm2d(16, False)(x)
        x = F.elu(x)
        x = nn.MaxPool2d(1, 4)(x)
        x = F.dropout(x, 0.25)

        # Layer 3
        x = nn.ZeroPad2d((8, 7, 0, 0))(x)
        x = nn.Conv2d(16, 16, (1, 16))(x)
        x = nn.BatchNorm2d(16, False)(x)
        x = F.elu(x)
        x = nn.MaxPool2d((1, 8))(x)
        x = F.dropout(x, 0.25)

        # 全连接层
        x = x.view(-1, 96)
        x = nn.Linear(96, 1)(x)  # 全连接
        # x = F.softmax(x, dim=1)  # 激活函数
        x = F.sigmoid(x)  # sigmoid激活函数
        return x
        print('forward done')

# 定义评估指标
def evaluate(model, X, Y, params=["acc"]):
    results = []
    batch_size = 100

    predicted = []

    for i in range(len(X) // batch_size):  # iteration,//向下取整
        s = i * batch_size
        e = i * batch_size + batch_size

        inputs = Variable(torch.from_numpy(X[s:e]))
        pred = model(inputs)

        predicted.append(pred.data.cpu().numpy())

    inputs = Variable(torch.from_numpy(X))
    predicted = model(inputs)
    predicted = predicted.data.cpu().numpy()
    """
    设置评估指标：
    acc：准确率
    auc:AUC 即 ROC 曲线对应的面积
    recall:召回率
    precision:精确率
    fmeasure：F值
    """
    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2 * precision * recall / (precision + recall))
    return results



def main():
    # 构建网络结构EEGNet,并设置二分类交叉熵和Adam优化器
    # 定义网络
    net = EEGNet()
    # 定义二分类交叉熵 (Binary Cross Entropy)
    criterion = nn.BCELoss()
    # 定义Adam优化器
    optimizer = optim.Adam(net.parameters())


    # 创建数据集
    """
    生成训练数据集，数据集有100个样本
    训练数据X_train:为[0,1)之间的随机数;
    标签数据y_train:为0或1
    """
    X_train = np.random.rand(300, 1, 62, 200).astype('float32')
    y_train = np.round(np.random.rand(300).astype('float32'))

    # datapath = os.getcwd()  # 获取当前文件路径
    # train_data = datapath + '\\data\\source1_15_2_300.mat'
    # train_label = datapath + '\\data\\label1_15_2_300.mat'
    # source_data = scipy.io.loadmat(train_data)
    # source_label = scipy.io.loadmat(train_label)
    # X_train = source_data['eegdata']
    # y_train = source_label['eeglabel']
    # print('load train data done')

    """
    生成验证数据集，数据集有100个样本
    验证数据X_val:为[0,1)之间的随机数;
    标签数据y_val:为0或1
    """
    X_val = np.random.rand(300, 1, 62, 200).astype('float32')
    y_val = np.round(np.random.rand(300).astype('float32'))

    """
    生成测试数据集，数据集有100个样本
    测试数据X_test:为[0,1)之间的随机数;
    标签数据y_test:为0或1
    """
    X_test = np.random.rand(300, 1, 62, 200).astype('float32')
    y_test = np.round(np.random.rand(300).astype('float32'))

    # tset_data = datapath + '\\data\\source1_15_2_300.mat'
    # test_label = datapath + '\\data\\label1_15_2_300.mat'
    # target_data = scipy.io.loadmat(tset_data)
    # target_label = scipy.io.loadmat(test_label)
    # X_test = target_data['eegdata']
    # y_test = target_label['eeglabel']
    # print('load test data done')

    # 训练并验证
    batch_size = 64
    # 训练 循环 10次
    for epoch in range(10):
        print("\nEpoch ", epoch)

        running_loss = 0.0
        for i in range(len(X_train) // batch_size - 1):
            s = i * batch_size
            e = i * batch_size + batch_size

            inputs = torch.from_numpy(X_train[s:e])
            labels = torch.FloatTensor(np.array([y_train[s:e]]).T * 1.0)

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        # 验证
        params = ["acc", "auc", "fmeasure"]
        print(params)
        print("Training Loss ", running_loss)
        print("Train - ", evaluate(net, X_train, y_train, params))
        print("Validation - ", evaluate(net, X_val, y_val, params))
        print("Test - ", evaluate(net, X_test, y_test, params))


if __name__ == '__main__':
    main()





