
import scipy.io
import os
import numpy as np
import csv


def main():
    X_train = np.random.rand(100, 1, 128, 64).astype('float32')
    y_train = np.round(np.random.rand(100).astype('float32'))
    print(y_train.shape)

    datapath = os.getcwd()  # 获取当前文件路径
    train_data = datapath + '\\data\\1_2_200.mat'
    train_label = datapath + '\\data\\label1_2.mat'
    source_data = scipy.io.loadmat(train_data)
    source_label = scipy.io.loadmat(train_label)
    X_train = source_data['eegdata']
    y_train = source_label['trainLabel'].T

    # X_train = X_train[:,0:X_train.shape[1]-1:5]  # resample:200Hz
    # mat_path = datapath + '\\data\\1_2_200.mat'
    # scipy.io.savemat(mat_path,{'eegdata':X_train})



    # 获取每个 trail 的时间节点信息
    info = datapath + '\\data\\info.csv'
    with open(info,'r') as f:
        reader = csv.reader(f)
        time_info = list(reader)
    start_points = time_info[2]
    end_points = time_info[3]
    start_points = [i for i in start_points if i != '']  # 删除列表空元素
    n_trails = len(start_points)

    print(start_points[0])
    # 重新拼接每个trail的视频段数据
    tempEEG = np.empty([62,0])
    print(type(tempEEG))
    print(type(X_train))
    for i in range(n_trails):
        start = int(start_points[i])*200
        end = (int(end_points[i])+1)*200
        tempEEG = np.hstack((tempEEG, X_train[:, start:end]))
    X_train = tempEEG

    # 将数据转换成 num*62*200 形式
    col = X_train.shape[1]
    n = X_train.shape[1] // 200  # 样本数
    X_train = X_train[:, 0:n * 200]
    X_train = X_train.reshape(n, 62, 200)
    # num*62*200

    print('1......')


if __name__ == '__main__':
    main()