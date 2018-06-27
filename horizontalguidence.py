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
    info = guidance(cav, net, tht_direction='strategy')
    states, ws, hcmds, thts = info['state_records'], info['w_records'], info['hcmd_records'], info['tht_records']
    ts, angles = info['ts'], info['angles']
    # 纵向制导画图
    # 单个HV走廊
    fig1, ax1 = plt.subplots()
    cav.plot(states, ax=ax1)
    ax1.set_xlabel('V(m/s)')
    ax1.set_ylabel('H(km)')
    # 单个经纬度图
    fig2, ax2 = plt.subplots()
    rate = 180 / math.pi
    ax2.plot(states[:, 1] * rate, states[:, 2] * rate)
    ax2.scatter([cav.gama0 * rate, cav.gamaf * rate], [cav.phi0 * rate, cav.phif * rate], marker='*', c='r')
    ax2.grid()
    ax2.set_xlabel('Longitude($^\circ$)')
    ax2.set_ylabel('Latitude($^\circ$)')
    # 单个权重系数变化图
    fig3, ax3 = plt.subplots()
    ax3.plot(ts, ws)
    ax3.set_xlabel('Time(s)')
    ax3.set_ylabel('Weight')
    ax3.set_ylim([0, 1.5])
    # 单个倾侧角度变化图
    fig4, ax4 = plt.subplots()
    ax4.plot(ts[:-5], np.abs(thts[:-5]))
    ax4.set_xlabel('Time(s)')
    ax4.set_ylabel('Angle($^\circ$)')

    print(info['range_error'])
    # cav = AircraftEnv()
    # # 进行飞行
    # info = guidance(cav, net, tht_direction='strategy')
    # states, ws, hcmds, thts = info['state_records'], info['w_records'], info['hcmd_records'], info['tht_records']
    # ax1.plot(states[:, 3], states[:, 0] / 1000 - cav.R0)

    plt.show()
