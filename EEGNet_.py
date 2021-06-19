# 2020.7.2

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
# torch.nn.functional.conv1d
# torch.nn.functional.conv_transpose1d
# ......functional. 还有很多其他函数
import torch.optim as optim
import os
import scipy.io

# 设置相关参数
channels = 62*5
timepoints = 200

kernels, chans, samples = 1, 62*5, 1

# 定义网络模型
class EEGNet(nn.Module):
    def __init__(self):  # 初始化网络架构
        super(EEGNet, self).__init__()
        self.T = 120

        # Layer 1
        # self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.conv1 = nn.Conv2d(1, 16, (1, 310), padding=0)
        # 2维卷积，输入通道数1，输出通道数16，卷积核大小1*64，零填充
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
        # self.fc1 = nn.Linear(4 * 2 * 7, 1)  # 对输入数据进行一个线性变换
        self.fc1 = nn.Linear(4 * 2 * 7, 1)

    def forward(self, x):  # 前向传播，由输入获取输出的一个过程
        # 输入x是一个张量
        # Layer 1
        x = F.elu(self.conv1(x))
        # torch.nn.functional.elu 非线性激活函数
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)  # 缓解过拟合
        x = x.permute(0, 3, 1, 2)  # 将tensor的维度换位
        print('layer1 forward done')


        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        print('layer2 forward done')

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        print('layer3 forward done')

        # 全连接层
        x = x.view(-1, 4 * 2 * 7)
        x = F.sigmoid(self.fc1(x))  # sigmoid激活函数
        return x

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
    # X_train = np.random.rand(100, 1, 120, 64).astype('float32')
    # y_train = np.round(np.random.rand(100).astype('float32'))

    datapath = os.getcwd()  # 获取当前文件路径
    train_data = datapath + '\\data\\source1_15_2_300.mat'
    train_label = datapath + '\\data\\label1_15_2_300.mat'
    source_data = scipy.io.loadmat(train_data)
    source_label = scipy.io.loadmat(train_label)
    X_train = source_data['eegdata'].T
    y_train = source_label['eeglabel'].T.astype('float32')
    # X_train = X_train.reshape((X_train.shape[0], 1, 1, X_train.shape[1]))
    # X_train = X_train.view([X_train.shape[0], 1, 200, X_train.shape[1]])
    X_train = np.expand_dims(X_train, axis=0)
    X_train = np.expand_dims(X_train, axis=0)
    X_train = X_train.swapaxes(2, 0).astype('float32')
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    print('load train data done')



    """
    生成验证数据集，数据集有100个样本
    验证数据X_val:为[0,1)之间的随机数;
    标签数据y_val:为0或1
    """
    # X_val = np.random.rand(100, 1, 120, 64).astype('float32')
    # y_val = np.round(np.random.rand(100).astype('float32'))

    """
    生成测试数据集，数据集有100个样本
    测试数据X_test:为[0,1)之间的随机数;
    标签数据y_test:为0或1
    """
    # X_test = np.random.rand(100, 1, 120, 64).astype('float32')
    # y_test = np.round(np.random.rand(100).astype('float32'))

    test_data = datapath + '\\data\\source1_15_2_300.mat'
    test_label = datapath + '\\data\\label1_15_2_300.mat'
    target_data = scipy.io.loadmat(test_data)
    target_label = scipy.io.loadmat(test_label)
    X_test = target_data['eegdata'].T
    y_test = target_label['eeglabel'].T.astype('float32')
    # X_test = X_test.reshape(X_test.shape[0], 1, 200, X_test.shape[1])
    X_test = np.expand_dims(X_test, axis=0)
    X_test = np.expand_dims(X_test, axis=0)
    X_test = X_test.swapaxes(2, 0).astype('float32')
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    print('load test data done')

    # 训练并验证
    batch_size = 256
    # 训练 循环 10次
    for epoch in range(2):
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
        # print("Validation - ", evaluate(net, X_val, y_val, params))
        print("Test - ", evaluate(net, X_test, y_test, params))


if __name__ == '__main__':
    main()

