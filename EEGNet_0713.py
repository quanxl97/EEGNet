# 2020.7.14

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
timepoints = 128

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120

        # layer1




if __name__ == '__main__':
    # datapath = os.getcwd()  # 获取当前文件路径
    # train_data = datapath + '\\data\\source1_15_2_300.mat'
    # train_label = datapath + '\\data\\label1_15_2_300.mat'
    # source_data = scipy.io.loadmat(train_data)
    # source_label = scipy.io.loadmat(train_label)
    # X_train = source_data['eegdata'].T
    # y_train = source_label['eeglabel'].T.astype('float32')
    # X_train = np.expand_dims(X_train, axis=0)
    # X_train = np.expand_dims(X_train, axis=0)
    # X_train = X_train.swapaxes(2, 0).astype('float32')

    X_train = np.random.rand(300, 1, 62, 200).astype('float32')
    y_train = np.round(np.random.rand(300).astype('float32'))

    print('load train data done')

    s, e = 0, 32  # 选取128条数据
    inputs = torch.from_numpy(X_train[s:e])
    labels = torch.FloatTensor(np.array([y_train[s:e]]).T * 1.0)
    # 将数组转化成tensor

    # wrap them in Variable
    inputs, labels = Variable(inputs), Variable(labels)
    # 对tensor进行封装
    print(inputs.shape)

    # layer 1
    # x = nn.Conv2d(1, 8, (1, 64), padding=0)(inputs)
    # x = F.pad(inputs, (32, 31, 0, 0))
    x = nn.ZeroPad2d((50, 49, 0, 0))(inputs)
    x = nn.Conv2d(1, 8, (1, 100), padding=0)(x)
    print(x.shape)
    x = nn.BatchNorm2d(8, False)(x)  # 数据做一个归一化
    print(x.shape)
    print('layer1 done')

    # Layer 2
    # x = nn.ZeroPad2d((16, 17, 0, 1))(x)
    print(x.shape)
    x = nn.Conv2d(8, 16, (62, 1))(x)
    print(x.shape)
    x = nn.BatchNorm2d(16, False)(x)
    print(x.shape)
    x = F.elu(x)
    print(x.shape)
    x = nn.MaxPool2d(1, 4)(x)
    print(x.shape)
    x = F.dropout(x, 0.25)
    print(x.shape)
    print('layer2 done')

    # Layer 3
    x = nn.ZeroPad2d((8, 7, 0, 0))(x)
    x = nn.Conv2d(16, 16, (1, 16))(x)
    print(x.shape)
    print('Conv2d done')
    x = nn.BatchNorm2d(16, False)(x)
    x = F.elu(x)
    print(x.shape)
    x = nn.MaxPool2d((1, 8))(x)
    print(x.shape)
    x = F.dropout(x, 0.25)
    print(x.shape)
    print('layer3 done')

    # flatten = nn.Flatten(x)

    # 全连接层
    x = x.view(-1, 96)
    print(x.shape)
    print('view done')
    x = nn.Linear(96, 1)(x)  # 全连接
    print(x.shape)
    x = F.softmax(x, dim=1)  # 激活函数
    print(x.shape)
    print('layer3 forward done')


