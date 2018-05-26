import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import root
import scipy.io as sio


class CAV:
    """
    CAV的飞行参数与气动模型
    """

    def __init__(self, low=True):
        # 飞行器总体参数,一般来说不用变
        self.m0 = 907  # kg
        self.S = 0.35  # m2
        self.R0 = 6378  # Km
        self.g0 = 9.8  # m/s^2
        self.Rd = 0.1
        self.C1 = 11030
        self.Vc = (self.g0 * self.R0 * 1000) ** 0.5
        self.Tc = (self.R0 * 1000 / self.g0) ** 0.5
        self.beta = 7200
        self.rho0 = 1.225
        self.low = low

    def get_x_dot(self, x, alpha, tht, p=0):
        # 对当前状态求导，得到状态导数
        # 输入当前状态、攻角(度)、倾侧角(度)、推力（沿机体坐标系x），输出状态导数
        # 状态提取[半径m，维度，经度，速度m/s，弹道倾角(弧度)，弹道偏角(弧度)]
        r, gamma, phi, v, theta, chi, range = x

        # 环境模型
        h = r - self.R0 * 1000  # 高度
        g = self.h2g(h)
        rho = self.h2rho(h)  # 密度

        # 气动力参数
        cl, cd = self.alphama2clcd(alpha, v / 340)  # 升力阻力系数
        q = 0.5 * rho * v ** 2  # 动压
        d = q * cl * self.S  # 阻力
        l = q * cd * self.S  # 升力
        ny = math.sqrt((p * math.sin(alpha / 57.3) + l) ** 2 +
                       (p * math.cos(alpha / 57.3) - d) ** 2) / (self.m0 * self.g0)  # 过载

        # 运动方程
        r_dot = v * math.sin(theta)
        gama_dot = v * math.cos(theta) * math.sin(chi) / r / math.cos(phi)  # y射程
        phi_dot = v * math.cos(theta) * math.cos(chi) / r  # x射程
        v_dot = p * math.cos(alpha / 57.3) / self.m0 - d / self.m0 - g * math.sin(theta)
        theta_dot = 1 / v * ((l + p * math.sin(alpha / 57.3)) * math.cos(tht / 57.3) / self.m0 -
                             (g - v ** 2 / r) * math.cos(theta))
        chi_dot = 1 / v * ((l + p * math.sin(alpha / 57.3)) * math.sin(tht / 57.3) / math.cos(theta) / self.m0 +
                           (v ** 2 / r) * math.cos(theta) * math.sin(chi) * math.tan(phi))
        range_dot = v * math.cos(theta) / r  # 速度向平面投影积分射程
        Q_dot = self.C1 / math.sqrt(self.Rd) * (rho / self.rho0) ** 0.5 * (v / self.Vc) ** 3.15
        info = {"Q_dot": Q_dot, "ny": ny, "q": q / 1000, "tht": tht, "alpha": alpha}

        x_dot = np.array([r_dot, gama_dot, phi_dot, v_dot, theta_dot, chi_dot, range_dot])
        return x_dot, info

    def v2alpha(self, v):
        # 输入速度m/s,输出攻角(度),这是CAV_L本身的设计决定的,不需要改变
        v1 = 3100
        v2 = 4700
        alpha_max = 20
        alpha_min = 8.5
        if v < v1:
            alpha = alpha_min
        elif v < v2:
            alpha = alpha_min + (alpha_max - alpha_min) / (v2 - v1) * (v - v1)
        else:
            alpha = alpha_max
        return alpha

    def h2g(self, h):
        # h的单位为m,物理规律,不需要改变
        g = self.g0 * (self.R0 * 1000) ** 2 / (self.R0 * 1000 + h) ** 2
        return g

    def h2rho(self, h):
        # h的单位为m,物理规律，不需要改变
        rho = self.rho0 * math.exp(-h / self.beta)
        return rho

    def alphama2clcd(self, alpha, ma):
        # 攻角(度)马赫数转换为升力系数、阻力系数
        if self.low == True:
            # 林哥毕业设计使用的是CAV-L
            # 低升阻比数据 cav-l数据，最大升阻比为2.4
            p00 = -0.2892
            p10 = 0.07719
            p01 = 0.03159
            p20 = -0.0006198
            p11 = -0.007331
            p02 = 0.0003381
            p21 = 8.544e-05
            p12 = 0.0003569
            p03 = -0.0001313
            p22 = -2.406e-06
            p13 = -5.55e-06
            p04 = 3.396e-06
            cl = p00 + p10 * alpha + p01 * ma + p20 * alpha ** 2 + p11 * alpha * ma + p02 * ma ** 2 + \
                 p21 * alpha ** 2 * ma + p12 * alpha * ma ** 2 + p03 * ma ** 3 + p22 * alpha ** 2 * ma ** 2 \
                 + p13 * alpha * ma ** 3 + p04 * ma ** 4
            p00 = 0.3604
            p10 = -0.02188
            p01 = -0.07648
            p20 = 0.001518
            p11 = 0.00450
            p02 = 0.0058
            p21 = -0.0001462
            p12 = -0.0001754
            p03 = -0.0002027
            p22 = 4.783e-06
            p13 = 1.025e-06
            p04 = 2.99e-06
            cd = p00 + p10 * alpha + p01 * ma + p20 * alpha ** 2 + p11 * alpha * ma + p02 * ma ** 2 + \
                 p21 * alpha ** 2 * ma + p12 * alpha * ma ** 2 + p03 * ma ** 3 + p22 * alpha ** 2 * ma ** 2 \
                 + p13 * alpha * ma ** 3 + p04 * ma ** 4
        else:
            # 高升阻比
            # 最大升阻比3.5
            cl = -0.0196 - 0.02065 * ma + 0.002347 * ma ** 2 - (8.486e-05) * ma ** 3 + (9.987e-07) * ma ** 4 + \
                 0.05097 * alpha - 0.0009715 * alpha * ma - (3.141e-06) * alpha * ma ** 2 + \
                 (4.148e-07) * alpha * ma ** 3 + 0.0004383 * alpha ** 2 - (9.948e-06) * alpha ** 2 * ma + \
                 (2.897e-07) * alpha ** 2 * ma ** 2
            cd = 0.4796 - 0.07427 * ma + 0.003595 * ma ** 2 - (7.542e-05) * ma ** 3 + (1.029e-06) * ma ** 4 + \
                 (-0.04037) * alpha + 0.005605 * alpha * ma - 0.0001142 * alpha * ma ** 2 + \
                 (-2.286e-06) * alpha * ma ** 3 + 0.002492 * alpha ** 2 - (0.0002678) * alpha ** 2 * ma + \
                 (8.114e-06) * alpha ** 2 * ma ** 2
        return cl, cd

    def phigamma2range(self, gamma0, phi0, gamma1, phi1):
        # 给出两点的经纬度计算射程
        ad = 2 * math.cos(phi0) * math.sin((gamma1 - gamma0) / 2)
        ec = 2 * math.cos(phi1) * math.sin((gamma1 - gamma0) / 2)
        cd = 2 * math.sin((phi1 - phi0) / 2)
        ac = (ad * ec + cd ** 2) ** 0.5
        range = 2 * math.asin(ac / 2) * self.R0
        return range


class AircraftEnv(CAV):
    """
    编写飞行器Markov模型
    分为几个部分
    x 表示内部飞行器所有信息
    state 外部所需要的信息
    """

    def __init__(self, random=False, low=True):
        CAV.__init__(self, low)
        self.x = None  # 内部循环状态变量
        self.state = None  # 深度强化学习中或者与外界交互变量，长度小于x
        self.t = 0
        self.delta_t = 1  # 积分步长
        self.random = random

        # 飞行器初始位置信息
        self.h0 = 100  # Km
        self.r0 = self.R0 + self.h0  # Km
        self.v0 = 7200  # m/s
        self.theta0 = -2 / 180 * math.pi  # 弧度  弹道倾角
        self.chi0 = 55 / 180 * math.pi  # 弧度    弹道偏角
        self.gama0 = 160 / 180 * math.pi  # 弧度  经度
        self.phi0 = 5 / 180 * math.pi  # 弧度     维度
        self.range0 = 0  # Km   射程
        self.Q0 = 0
        self.Osci0 = 0
        self.x0 = np.hstack((self.r0 * 1000, self.gama0, self.phi0, self.v0, self.theta0, self.chi0, self.range0))

        # 飞行器终点位置 终端位置高度速度射程
        self.hf = 20  # km
        self.rf = self.R0 + self.hf  # km
        self.vf = 1800  # m/s
        self.gamaf = 225 / 180 * math.pi  # 弧度
        self.phif = 25 / 180 * math.pi  # 弧度
        self.rangef = self.phigamma2range(self.gama0, self.phi0, self.gamaf, self.phif)  # 射程km

        # 约束条件
        self.Q_dot_max_allow = 1200  # Kw/m2
        self.n_max_allow = 4  # g0
        self.q_max_allow = 200  # Kpa

        # 进行初始化
        self.reset()

        # 环境维度
        self.s_dim = len(self.state)
        self.a_dim = None

    def _x2state(self, x):
        # 将x缩减为state
        return x.copy()

    def render(self):
        pass

    def reset(self):
        self.t = 0
        if self.random:
            pass
        else:
            self.x = self.x0.copy()
        self.state = self._x2state(self.x)
        return self.state.copy()

    def h_v(self, tht):
        # 构建HV走廊
        v = np.linspace(self.v0, self.vf, 1001)  # 速度 m/s
        # h的下界值，h太小密度热会超约束
        h_Qmax = v.copy()
        h_nmax = v.copy()
        h_qmax = v.copy()
        h_down = v.copy()
        # h的上界值，h太高会掉下来
        h_qegc = v.copy()  # 平衡滑翔条件

        def f(h, vv):
            g = self.h2g(h)
            rho = self.h2rho(h)
            r = self.R0 * 1000 + h
            q = 0.5 * rho * vv ** 2  # 动压
            l = q * cl * self.S  # 升力
            return l * math.cos(tht) / self.m0 + (vv ** 2 / r) - g

        for i in range(len(v)):
            # 计算三个等式
            h_Qmax[i] = 2 * self.beta * math.log((self.C1 * (v[i] / self.Vc) ** 3.15) /
                                                 (self.Q_dot_max_allow * (self.Rd ** 0.5))) / 1000
            alpha = self.v2alpha(v[i])
            cl, cd = self.alphama2clcd(alpha, v[i] / 340)
            h_nmax[i] = 1 * self.beta * math.log((self.rho0 * v[i] ** 2 * self.S * (cl ** 2 + cd ** 2) ** 0.5) /
                                                 (2 * self.n_max_allow * self.m0 * self.g0)) / 1000
            h_qmax[i] = 1 * self.beta * math.log((self.rho0 * v[i] ** 2) / (2 * self.q_max_allow * 1000)) / 1000

            h_down[i] = max(h_Qmax[i], h_nmax[i], h_qmax[i])
            # 计算平衡滑翔的对应的高度
            res = root(f, 0, (v[i],))
            h_qegc[i] = res.x / 1000
        fig = plt.figure()
        plt.plot(v, h_Qmax, v, h_nmax, v, h_qmax, v, h_qegc, v, h_down)

        plt.legend(['h_Q_dot', 'h_n', 'h_q', 'h_qegc', 'h_down'])
        plt.grid()

    def step(self, action):
        # 假设倾侧角是action度 正负90度范围
        # 使用固定序列的攻角
        v = self.x[3]
        alpha = self.v2alpha(v)
        # 简单积分
        x_dot, info = self.get_x_dot(self.x, alpha, action)
        self.x += x_dot * self.delta_t

        done = None

        self.state = self._x2state(self.x)

        reward = None

        info = info.copy()
        return self.state, reward, done, info

    def plot(self, data):
        # 画出轨迹的图，输入数据每行代表某一个时刻的状态量
        t = np.arange(0, len(data), 1) * self.delta_t  # 时间
        fig = plt.figure()
        plt.plot(t, data[:, 3])


if __name__ == '__main__':
    cav = AircraftEnv()
    state_now = cav.reset()
    state_record = state_now.copy()
    for i in range(1500):
        action = np.random.rand(1) * 180 - 90
        state_now, reward, done, info = cav.step(action)
        state_record = np.vstack((state_record, state_now.copy()))  # 垂直添加
    cav.plot(state_record)
    cav.h_v(0 / 180 * math.pi)
    plt.show()
