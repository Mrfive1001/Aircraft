import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from dnn import DNN

# 对射程进行预测

if __name__ == '__main__':
    train_mode = 0  # 是否进行网络训练,0不训练，1从0开始训练，2从之前基础上开始训练
    num = 7000  # 使用所有点进行训练
    net = DNN(2, 1, 256, train=train_mode, isnorm=True, name='range_%s' % str(num))  # 定义网络
    memory = np.load('Trajectories/memory_%s.npy' % str(num))  # 读取数据
    memory_norm = net.norm(memory)
    if train_mode == 1 or train_mode == 2:
        # 训练模式
        X = memory_norm[:, [0, 1]].copy()
        Y = memory_norm[:, 2:].copy()
        losses = []
        for i in range(40000):
            sample_index = np.random.choice(len(X), size=500)
            batch_x = X[sample_index, :]
            batch_y = Y[sample_index, :]
            loss, mae = net.learn(batch_x, batch_y)
            losses.append(loss)
            print(i + 1, '射程预测平均误差是', mae * 100, 'km')
        plt.plot(losses)
        sample_index = np.random.choice(len(X), size=100)
        batch_x = X[sample_index, :]
        batch_y = Y[sample_index, :]
        batch_y_pre = net.predict(batch_x)
        error_y = net.unorm(np.hstack((batch_x, batch_y - batch_y_pre)))[:, -1]
        print(np.mean(np.abs(error_y)))
        net.store()
        plt.show()
    else:
        X = memory_norm[:, [0, 1]].copy()
        Y = memory_norm[:, 2:].copy()
        tes_num = 100
        sample_index = np.random.choice(len(X), size=tes_num)
        batch_x = X[sample_index, :]
        batch_y = Y[sample_index, :]
        batch_y_pre = net.predict(batch_x)
        for i in range(tes_num):
            print("第%d个数据，w = %f,v = %f,range_real = %f,range_pre = %f,delta_range = %f" % (
                i + 1, batch_x[i, 0] * 0.01, batch_x[i, 1] * 100, batch_y[i, 0] * 100, batch_y_pre[i, 0] * 100,
                (batch_y[i, 0] - batch_y_pre[i, 0]) * 100))
        print('射程预测平均误差是', np.mean(batch_y - batch_y_pre) * 100, 'km')
