import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from aircraft_obj import AircraftEnv
from scipy.optimize import root
from scipy.interpolate import interp1d
from dnn import DNN

# 对w进行预测并且使用跟踪

if __name__ == '__main__':
    train_mode = 0  # 是否进行网络训练,0不训练，1从0开始训练，2从之前基础上开始训练
    num = 7000
    g = tf.Graph()
    memory = np.load('Trajectories/memory_%s.npy' % num)  # 读取数据
    net = DNN(2, 1, 512, train=train_mode, isnorm=True, name='w_%s' % str(num), graph=g, scale=[0.1, 100, 100])  # 定义网络
    memory_norm = net.norm(memory)
    if train_mode != 0:
        # 训练模式
        X = memory_norm[:, 1:].copy()
        Y = memory_norm[:, 0:1].copy()
        losses = []
        for i in range(5000):
            sample_index = np.random.choice(len(X), size=1000)
            batch_x = X[sample_index, :]
            batch_y = Y[sample_index, :]
            loss, mae = net.learn(batch_x, batch_y)
            losses.append(loss)
            print(i + 1, '预测平均误差是', mae)
        plt.plot(losses)
        net.store()
        plt.show()
    else:
        cav = AircraftEnv()
        # 初始设置
        state_now = cav.reset()
        cons = 22
        v2h_down = interp1d(cav.vv, cav.h_down, kind='quadratic')
        v2h_up = interp1d(cav.vv, cav.h_up, kind='quadratic')
        range_target = 7000
        v_init = 7000
        v = state_now[3]
        range_now = state_now[-1] * cav.R0
        range_left = range_target - range_now
        state_record = state_now.copy()
        ws = []
        h_cmds = []
        thts = []
        # 开始测试
        while True:
            # 得到所需w
            if v > v_init:
                tht = cons / 57.3
                w = 1
                h_cmd = (1 - w) * v2h_down(min(v, cav.v0)) + w * v2h_up(min(v, cav.v0))  # m
            else:
                # 对高度进行跟踪
                w = net.predict(net.norm([[1, v, range_left]])[:, 1:])[0] / 10
                # 计算目标高度
                h_cmd = (1 - w) * v2h_down(min(v, cav.v0)) + w * v2h_up(min(v, cav.v0))  # m
                tht = cav.h2tht(h_cmd, h_cmds)
            ws.append(w)
            h_cmds.append(h_cmd)
            thts.append(tht)
            state_now, reward, done, info = cav.step(tht * 57.3)
            # 计算当前据目标射程
            v = state_now[3]
            range_now = state_now[-1] * cav.R0
            range_left = range_target - range_now
            state_record = np.vstack((state_record, state_now.copy()))  # 垂直添加
            if done:
                h_cmds.append(h_cmds[-1])
                break
        print(range_left)
        cav.plot(state_record, h_cmds)
        plt.figure()
        plt.plot(ws)
        plt.show()
