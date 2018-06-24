import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from aircraft_obj import AircraftEnv
from dnn import DNN
from verticalguidence import guidance
from scipy.optimize import root
from scipy.interpolate import interp1d
import math

# 横纵向弹道 横向确定方向，纵向确定大小
if __name__ == '__main__':
    # 纵向网络的定义
    num = 7000
    net = DNN(2, 1, 512, train=0, isnorm=True, name='w_%s' % str(num), scale=[0.1, 100, 100])  # 定义网络
    # 定义训练对象
    cav = AircraftEnv()
    # 进行飞行
    info = guidance(cav,net,tht_direction='strategy')
    states, ws, hcmds, thts = info['state_records'], info['w_records'], info['hcmd_records'], info['tht_records']
    rate = 180 / math.pi
    plt.figure()
    plt.plot(states[:, 1] * rate, states[:, 2] * rate)
    print(info['target_error'])
    plt.plot([cav.gama0 * rate, cav.gamaf * rate], [cav.phi0 * rate, cav.phif * rate], marker='*', c='r')
    plt.grid()
    plt.xlabel('Longitude/$^\circ$')
    plt.xlabel('Latitude/$^\circ$')
    plt.show()
