"""
2020.7.14
网络输入采用200Hz降采样+滤波+PSD特征提取的数据
通道数：62
特征维数：62*5
!!! 这样会出现问题，与EEGNet的原理会不一样，不过还不知道后面的结果会是怎么样的
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
import csv
import random
import numpy
from scipy import signal

channels = 62
timepoints = 200
fs_sample = 1000
fs_resample = 200


# 定义网络模型
class EEGNet(nn.Module):
    def __init__(self):  # 初始化网络架构
        super(EEGNet, self).__init__()
        self.T = 120
        '''
        resample: 200Hz
        '''
        # Layer 1
        self.padding1 = nn.ZeroPad2d((50, 49, 0, 0))
        self.conv1 = nn.Conv2d(1, 8, (1, 100), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(8, False)

        # Layer 2
        self.conv2_1 = nn.Conv2d(8, 8, (62, 1), groups=8, bias=False)  # 二维卷积
        self.conv2_2 = nn.Conv2d(8, 16, (1, 1), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16, False)
        self.pooling2 = nn.MaxPool2d(1, 4)
        # 对于输入信号的输入通道，提供2维最大池化操作

        # Layer 3
        self.padding3 = nn.ZeroPad2d((8, 7, 0, 0))
        self.conv3 = nn.Conv2d(16, 16, (1, 16))
        self.conv3_1 = nn.Conv2d(16, 16, (1, 16), groups=16, bias=False)  # 二维卷积
        # self.conv3_2 = nn.Conv2d(8, 16, (1, 1), bias=False)

        self.batchnorm3 = nn.BatchNorm2d(16, False)
        self.pooling3 = nn.MaxPool2d((1, 8))

        # 全连接层
        # 此维度将取决于数据中每个样本的时间戳数。
        # I have 120 timepoints.
        self.fc1 = nn.Linear(96, 3)  # 对输入数据进行一个线性变换

        '''
        resample:128Hz
        '''
        # # Layer 1
        # self.padding1 = nn.ZeroPad2d((32, 31, 0, 0))
        # self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=0)
        # self.batchnorm1 = nn.BatchNorm2d(8, False)
        #
        # # Layer 2
        # self.conv2_1 = nn.Conv2d(8, 8, (62, 1), groups=8, bias=False)  # 二维卷积
        # self.conv2_2 = nn.Conv2d(8, 16, (1, 1), bias=False)
        # self.batchnorm2 = nn.BatchNorm2d(16, False)
        # self.pooling2 = nn.MaxPool2d(1, 4)
        # # 对于输入信号的输入通道，提供2维最大池化操作
        #
        # # Layer 3
        # self.padding3 = nn.ZeroPad2d((8, 7, 0, 0))
        # self.conv3 = nn.Conv2d(16, 16, (1, 16))
        # self.conv3_1 = nn.Conv2d(16, 16, (1, 16), groups=16, bias=False)  # 二维卷积
        # # self.conv3_2 = nn.Conv2d(8, 16, (1, 1), bias=False)
        #
        # self.batchnorm3 = nn.BatchNorm2d(16, False)
        # self.pooling3 = nn.MaxPool2d((1, 8))
        #
        # # 全连接层
        # # 此维度将取决于数据中每个样本的时间戳数。
        # # I have 120 timepoints.
        # self.fc1 = nn.Linear(96, 3)  # 对输入数据进行一个线性变换


    def forward(self, x):  # 前向传播，由输入获取输出的一个过程
        # 输入x是一个张量
        # layer 1
        x = self.padding1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)  # 数据做一个归一化

        # Layer 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        # x = self.pooling2(x)
        # AveragePooling2D
        x = F.adaptive_avg_pool2d(x, (1, 50))  # 自适应池化，指定池化输出尺寸为 1 * 1
        x = F.dropout(x, 0.25)

        # Layer 3
        x = self.padding3(x)
        x = self.conv3_1(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        # x = self.pooling3(x)
        x = F.adaptive_avg_pool2d(x, (1, 6))
        x = F.dropout(x, 0.25)

        # 全连接层
        x = x.view(-1, 96)  # 4维张量转换成2维
        x = nn.Linear(96, 3)(x)  # 全连接
        # x = F.softmax(x, dim=1)  # 激活函数
        # x = F.sigmoid(x)  # sigmoid激活函数
        x = F.log_softmax(x)
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
    # predicted = torch.max(predicted, 1)
    predicted = predicted.data.cpu().numpy()
    predicted = np.argmax(predicted, axis=1)
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
            results.append(accuracy_score(Y, predicted))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, predicted))
        if param == "precision":
            results.append(precision_score(Y, predicted))
        if param == "fmeasure":
            precision = precision_score(Y, predicted)
            recall = recall_score(Y, predicted)
            results.append(2 * precision * recall / (precision + recall))
    return results


def main():
    # 构建网络结构EEGNet,并设置交叉熵损失函数和Adam优化器
    # 定义网络
    net = EEGNet()
    # 定义交叉熵损失函数
    # criterion = nn.CrossEntropyLoss()
    # criterion = F.nll_loss()
    # criterion = nn.MSELoss()
    # 定义Adam优化器
    optimizer = optim.Adam(net.parameters())


    """
    训练集
    """
    datapath = os.getcwd()  # 获取当前文件路径
    train_data = datapath + '\\data\\1_2_200.mat'
    train_label = datapath + '\\data\\label1_2.mat'
    source_data = scipy.io.loadmat(train_data)
    source_label = scipy.io.loadmat(train_label)
    X_train = source_data['eegdata']  # raweeg, eegdata
    y_train = source_label['trainLabel'].T.astype('float32')
    # 标签类别编号
    y_train[y_train == 1] = 2
    y_train[y_train == 0] = 1
    y_train[y_train == -1] = 0
    # y_train = y_train.flatten()  # 降到1维

    # 降采样到200Hz
    # X_train = X_train[:,0:X_train.shape[1]-1:5]  # resample:200Hz
    # mat_path = datapath + '\\data\\1_2_200.mat'
    # scipy.io.savemat(mat_path,{'eegdata':X_train})

    # 滤波器
    b, a = signal.butter(4, [2*0.3/fs_resample, 2*70/fs_resample], 'bandpass')  # 四阶巴特沃斯滤波器
    X_train = signal.filtfilt(b, a, X_train)

    # 获取每个 trail 的时间节点信息
    info = datapath + '\\data\\info.csv'
    with open(info, 'r') as f:
        reader = csv.reader(f)
        time_info = list(reader)
    start_points = time_info[2]
    end_points = time_info[3]
    start_points = [i for i in start_points if i != '']  # 删除列表空元素
    n_trails = len(start_points)

    # 重新拼接每个trail的视频段数据
    tempEEG = np.empty([62, 0])
    for i in range(n_trails):
        start = int(start_points[i]) * 200
        end = (int(end_points[i]) + 1) * 200
        tempEEG = np.hstack((tempEEG, X_train[:, start:end]))
    X_train = tempEEG

    # 将数据转换成 num*62*200 形式
    col = X_train.shape[1]
    n = X_train.shape[1] // 200  # 样本数
    X_train = X_train[:, 0:n * 200]
    X_train = X_train.reshape(n, 1, 62, 200).astype('float32')
    # num*62*200

    # 打乱数据顺序
    state = np.random.get_state()
    np.random.shuffle(X_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)


    # 训练并验证
    batch_size = 64
    # 训练 循环 10次
    for epoch in range(3):
        print("\nEpoch ", epoch)

        running_loss = 0.0
        running_correct = 0
        for i in range(len(X_train) // batch_size - 1):  # iteration
            s = i * batch_size
            e = i * batch_size + batch_size

            inputs = torch.from_numpy(X_train[s:e])
            # labels = y_train[s:e].T.flatten()
            # labels = torch.FloatTensor(np.array([y_train[s:e]]).T * 1.0)
            labels = torch.from_numpy(y_train[s:e]).squeeze().long()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            # optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            _,pred = torch.max(outputs.data,1)

            optimizer.zero_grad()
            loss = F.nll_loss(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            # running_loss += loss.data[0]
            running_correct += torch.sum(pred == labels.data)

        # 验证
        # params = ["acc", "auc", "fmeasure"]
        params = ["acc"]
        print(params)
        print("Training Loss ", running_loss)
        print("Train - ", evaluate(net, X_train, y_train, params))
        # print("Validation - ", evaluate(net, X_val, y_val, params))
        # print("Test - ", evaluate(net, X_test, y_test, params))



if __name__ == '__main__':
    main()





