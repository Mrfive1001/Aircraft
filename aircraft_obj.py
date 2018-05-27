import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.interpolate import interp1d
import scipy.io as sio
import os


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
        self.rangef = self.phigamma2range(self.gama0, self.phi0, self.gamaf, self.phif)  # 射程km 与 range*self.R0差不多

        # 约束条件
        self.Q_dot_max_allow = 1200  # Kw/m2
        self.n_max_allow = 4  # g0
        self.q_max_allow = 200  # Kpa

        # 读取HV走廊
        try:
            HV = np.load('corrior.npz')
            self.vv = HV['vv']
            self.h_down = HV['h_down']
            self.h_up = HV['h_up']
        except:
            pass
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

    def hv_w(self, w):
        # 输入一个w值自动规划出一条弹道[0~1] w越大，离上界越近
        # 返回弹道记录
        state_now = self.reset()
        state_record = state_now.copy()
        v2h_down = interp1d(self.vv, self.h_down, kind='quadratic')
        v2h_up = interp1d(self.vv, self.h_up, kind='quadratic')
        while True:
            v = self.x[3]
            h = (1 - w) * v2h_down(min(v, self.v0)) + w * v2h_up(min(v, self.v0))
            if v > 7000:
                tht = 0
            else:
                tht = self.v2tht(h * 1000, v) / math.pi * 180
            state_now, reward, done, info = self.step(tht)
            state_record = np.vstack((state_record, state_now.copy()))  # 垂直添加
            if done:
                break
        return state_record.copy()

    def v2tht(self, h, v):
        # h 单位m，v单位m/s
        g = self.h2g(h)
        rho = self.h2rho(h)
        r = self.R0 * 1000 + h
        q = 0.5 * rho * v ** 2  # 动压
        alpha = self.v2alpha(v)
        cl, cd = self.alphama2clcd(alpha, v / 340)
        l = q * cl * self.S  # 升力
        a = self.m0 * (g - v ** 2 / r) / l
        tht = math.acos(min(self.m0 * (g - v ** 2 / r) / l, 1))
        return tht

    def h_v(self):
        # 构建HV走廊
        # 对几个典型的走廊进行对比
        # tht[0,40] n_max[3,4] Q_max[1200,1300] q_max[200,240]
        thts = [0, 40 / 180 * math.pi]
        n_maxs = [3, 4]
        Q_maxs = [1200, 1300]
        q_maxs = [200, 400]

        v = np.linspace(self.v0, self.vf, 1001)  # 速度 m/s
        v_change = 2000  # 自己设定的临界速度来规划HV走廊
        # h的下界值，h太小密度热会超约束
        h_Qmax1 = v.copy()
        h_Qmax2 = v.copy()
        h_nmax1 = v.copy()
        h_nmax2 = v.copy()
        h_qmax1 = v.copy()
        h_qmax2 = v.copy()
        h_down = v.copy()  # 下界
        h_down_change = v.copy()  # 人为设定的末端下界
        # h的上界值，h太高会掉下来
        h_qegc1 = v.copy()  # 平衡滑翔条件
        h_qegc2 = v.copy()  # 平衡滑翔条件
        h_qegc_change = v.copy()  # 人为设定的末端上界

        def f(h, vv, tht):
            # h单位m v单位m/s
            g = self.h2g(h)
            rho = self.h2rho(h)
            r = self.R0 * 1000 + h
            q = 0.5 * rho * vv ** 2  # 动压
            l = q * cl * self.S  # 升力
            return l * math.cos(tht) / self.m0 + (vv ** 2 / r) - g

        for i in range(len(v)):
            # 计算三个等式
            for tht, n_max, Q_max, q_max, h_nmax, h_Qmax, h_qmax, h_qegc in zip(
                    thts, n_maxs, Q_maxs, q_maxs, [h_nmax1, h_nmax2], [h_Qmax1, h_Qmax2], [h_qmax1, h_qmax2],
                    [h_qegc1, h_qegc2]):
                h_Qmax[i] = 2 * self.beta * math.log((self.C1 * (v[i] / self.Vc) ** 3.15) /
                                                     (Q_max * (self.Rd ** 0.5))) / 1000
                alpha = self.v2alpha(v[i])
                cl, cd = self.alphama2clcd(alpha, v[i] / 340)
                h_nmax[i] = 1 * self.beta * math.log((self.rho0 * v[i] ** 2 * self.S * (cl ** 2 + cd ** 2) ** 0.5) /
                                                     (2 * n_max * self.m0 * self.g0)) / 1000
                h_qmax[i] = 1 * self.beta * math.log((self.rho0 * v[i] ** 2) / (2 * q_max * 1000)) / 1000

                h_down[i] = max(h_Qmax[i], h_nmax[i], h_qmax[i])
                # 计算平衡滑翔的对应的高度
                res = root(f, 0, (v[i], tht))
                h_qegc[i] = res.x / 1000
                # 将末端走廊规划到一点，减少一个末端约束
                if v[i] >= v_change:
                    h_down_change[i] = h_down[i]
                    h_qegc_change[i] = h_qegc[i]
                else:
                    h_down_change[i] = (h_down_change[i - 1] - self.hf) * (v[i] - self.vf) / \
                                       (v[i - 1] - self.vf) + self.hf
                    h_qegc_change[i] = (h_qegc_change[i - 1] - self.hf) * (v[i] - self.vf) / \
                                       (v[i - 1] - self.vf) + self.hf

        # np.savez('corrior_orginal', vv=v, h_down=h_down, h_up=h_qegc)
        # np.savez('corrior', vv=v, h_down=h_down_change, h_up=h_qegc_change)

        # 将其显示
        # 画图
        fig_HV = plt.figure()
        ax = fig_HV.add_subplot(111)
        plt.plot(v, h_qegc1, v, h_qegc2, '--', v, h_qmax1, v, h_qmax2, '-.', v, h_Qmax1,
                 v, h_Qmax2, ':', v, h_nmax1, v, h_nmax2, ':')
        plt.annotate('$Q_{max} = %s Kw/m^2$' % (Q_maxs[0]), (2503, -3.7), (2892, -7),  # 文字，图像中的坐标，文字的坐标
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$Q_{max} = %s Kw/m^2$' % (Q_maxs[1]), (2293, -9), (2792, -14.3),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$q_{max} = %s KPa$' % (q_maxs[0]), (6290, 34.7), (6700, 24),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$q_{max} = %s KPa$' % (q_maxs[1]), (6036, 28.9), (6610, 20),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$n_{max} = %s $' % (n_maxs[0]), (2373.3, 18.43), (2318, 4),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$n_{max} = %s $' % (n_maxs[1]), (1834.4, 12.9), (2004, 0),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$tht = %s^\circ$' % (thts[0]), (4519, 43.4), (4185, 55),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$tht = %.2s^\circ $' % (thts[1] / math.pi * 180), (3850, 36), (3900, 50),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        ax.set_xlabel(r'v(m/s)')
        ax.set_ylabel(r'h(km)')

        ax.grid()
        plt.savefig(os.path.join('Figures', 'HV走廊敏感性分析.png'))

    def step(self, action):
        # 假设倾侧角是action度 正负90度范围
        # 使用固定序列的攻角
        v = self.x[3]
        alpha = self.v2alpha(v)
        # 简单积分
        x_dot, info = self.get_x_dot(self.x, alpha, action)
        self.x += x_dot * self.delta_t
        # 判断结束，以速度为指标
        if self.x[3] <= self.vf:
            done = True
        else:
            done = False
        self.state = self._x2state(self.x)
        # 奖励函数
        reward = None
        info = info.copy()
        return self.state, reward, done, info

    def plot(self, data):
        # 画出轨迹的图，输入数据每行代表某一个时刻的状态量
        t = np.arange(0, len(data), 1) * self.delta_t  # 时间
        fig = plt.figure()
        # plt.plot(t, data[:, 3])
        # plt.plot(self.vv, self.h_up, self.vv, self.h_down)
        plt.plot(data[:, 3], data[:, 0] / 1000 - self.R0)
        plt.grid()


if __name__ == '__main__':
    cav = AircraftEnv()

    cav.h_v()  # 进行HV走廊敏感性分析
    plt.show()
    # TODO 保存轨迹并进行神经网络构建
