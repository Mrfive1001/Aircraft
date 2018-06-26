import numpy as np
import tensorflow as tf
import os
import sys
import math
import matplotlib.pyplot as plt
from aircraft_obj import AircraftEnv
from scipy.optimize import root
from scipy.interpolate import interp1d
from dnn import DNN


# 对w进行预测并且使用跟踪
def guidance(cav, net, tht_direction=None, range_target=None):
    """
    :param cav: 飞行器对象
    :param net: 网络对象
    :param range_target: 目标射程
    :return: 返回信息
    """
    # 初始设置
    if range_target is None:
        range_target = cav.rangef
    state_now = cav.reset()
    cons = 22
    v2h_down = interp1d(cav.vv, cav.h_down, kind='quadratic')
    v2h_up = interp1d(cav.vv, cav.h_up, kind='quadratic')
    v_init = 7000
    v = state_now[3]
    range_now = state_now[-1] * cav.R0
    range_left = range_target - range_now
    state_record = state_now.copy()
    ws = []
    wuses = []
    h_cmds = []
    thts = []
    direction_last = 600
    count = direction_last
    direction = np.random.randint(0, 2) * 2 - 1
    # 开始测试
    angles = []
    while True:
        # 得到所需w
        if v > v_init:
            tht = cons / 57.3
            w = 1
            w_use = w
            h_cmd = (1 - w_use) * v2h_down(min(v, cav.v0)) + w_use * v2h_up(min(v, cav.v0))  # m
        else:
            # 对高度进行跟踪
            w = net.predict(net.norm([[1, v, range_left]])[:, 1:])[0] / 10
            w = min(1, w)
            # 计算目标高度
            if ws:
                alpha = 0.9
                w_use = alpha * w + (1 - alpha) * ws[-1]
            else:
                w_use = w
            h_cmd = (1 - w_use) * v2h_down(min(v, cav.v0)) + w_use * v2h_up(min(v, cav.v0))  # m
            tht = cav.h2tht(h_cmd, h_cmds)
        angle = cav.calculate_angle() / math.pi * 180
        angles.append(angle)
        # 横向制导
        # TODO 横向制导新方案
        if tht_direction == 'random':
            if count <= 0:
                direction = np.random.randint(0, 2) * 2 - 1
                direction_last = direction_last
                count = direction_last
            count = count - 1
            tht = tht * direction
        elif tht_direction == 'neg':
            tht = -tht
        elif tht_direction == 'simple':
            if abs(angle) > 10:
                if angle > 0:
                    direction = 1
                else:
                    direction = -1
            tht = tht * direction
        elif tht_direction == 'strategy':
            if v > 6000:
                limit = 10
            elif v > 3500:
                limit = 10
            else:
                limit = max(0, (v - 1800) * (10 - 2) / (3500 - 1800) + 2)
            if abs(angle) > limit:
                if angle > 0:
                    direction = 1
                else:
                    direction = -1
            tht = tht * direction
        ws.append(w)
        wuses.append(w_use)
        h_cmds.append(h_cmd)
        thts.append(tht)
        state_now, reward, done, info = cav.step(tht * 57.3)
        # 计算当前据目标射程
        v = state_now[3]
        range_now = state_now[-1] * cav.R0
        # range_left = range_target - range_now
        range_left = cav.calculate_range()
        state_record = np.vstack((state_record, state_now.copy()))  # 垂直添加
        if done:
            h_cmds.append(h_cmds[-1])
            angles.append(angles[-1])
            thts.append(thts[-1])
            ws.append(ws[-1])
            break
    t = np.arange(0, len(state_record), 1) * cav.delta_t  # 时间
    info = {'state_records': state_record, 'w_records': np.array(ws), 'hcmd_records': np.array(h_cmds),
            'tht_records': np.array(thts), 'range_error': range_left, 'target_error': cav.calculate_range(),
            'angles': np.array(angles),'ts':t}
    return info


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
        # 测试模式
        cav = AircraftEnv()
        info = guidance(cav, net)
        print(info['range_error'])
        state_record, h_cmds = info['state_records'], info['hcmd_records']
        cav.plot(state_record, h_cmds)
        plt.show()
