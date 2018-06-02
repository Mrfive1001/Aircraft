import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from aircraft_obj import AircraftEnv
from scipy.optimize import root
from scipy.interpolate import interp1d
from dnn import DNN

if __name__ == '__main__':
    cav = AircraftEnv()

    net = DNN(2, 1, 256, train=False, isnorm=True)  # 定义网络
    memory = np.load('memory.npy')  # 读取数据
    memory_norm = net.norm(memory)

    state_now = cav.reset()
    state_record = state_now.copy()
    cons = 22
    v2h_down = interp1d(cav.vv, cav.h_down, kind='quadratic')
    v2h_up = interp1d(cav.vv, cav.h_up, kind='quadratic')
    range_target = 6000
    v_init = 7000
    h_cmds = []
    while True:
        v = state_now[3]
        range_now = state_now[-1]*cav.R0
        range_left = range_target - range_now
        w = net.predict(net.norm([[1, v, range_left]])[:, 1:])[0]
        h_cmd = (1 - w) * v2h_down(min(v, cav.v0)) + w * v2h_up(min(v, cav.v0))  # m
        if v > v_init:
            tht = cons / 57.3
        else:
            tht = cav.h2tht(h_cmd, h_cmds)
        h_cmds.append(h_cmd)
        state_now, reward, done, info = cav.step(tht * 57.3)
        state_record = np.vstack((state_record, state_now.copy()))  # 垂直添加
        if done:
            break
    print(range_left)
    cav.plot(state_record)
    plt.show()