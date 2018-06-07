import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from dnn import DNN

# 对射程进行预测

if __name__ == '__main__':
    train_mode = 1  # 是否进行网络训练,0不训练，1从0开始训练，2从之前基础上开始训练
    num = 7000
    net = DNN(2, 1, 256, train=train_mode, isnorm=True, name='range_%s' % str(num))  # 定义网络
    memory = np.load('Trajectories/memory_%s.npy' % str(num))  # 读取数据
    memory_norm = net.norm(memory)
    if train_mode == 1 or train_mode == 2:
        # 训练模式
        X = memory_norm[:, [0, 1]].copy()
        Y = memory_norm[:, 2:].copy()
        losses = []
        for i in range(5000):
            sample_index = np.random.choice(len(X), size=5000)
            batch_x = X[sample_index, :]
            batch_y = Y[sample_index, :]
            loss = net.learn(batch_x, batch_y)
            losses.append(loss)
            print(i + 1, loss)
        plt.plot(losses)
        sample_index = np.random.choice(len(X), size=10)
        batch_x = X[sample_index, :]
        batch_y = Y[sample_index, :]
        batch_y_pre = net.predict(batch_x)
        print(batch_y, '\n\n')
        print(batch_y_pre - batch_y)
        net.store()
        plt.show()
    else:
        X = memory_norm[:, [0, 1]].copy()
        Y = memory_norm[:, 2:].copy()
        sample_index = np.random.choice(len(X), size=10)
        batch_x = X[sample_index, :]
        batch_y = Y[sample_index, :]
        batch_y_pre = net.predict(batch_x)
        print(batch_y, '\n\n')
        print(batch_y_pre - batch_y)
