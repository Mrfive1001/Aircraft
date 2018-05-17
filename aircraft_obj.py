import numpy as np
import math
import matplotlib.pyplot as plt


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

        # 约束条件
        self.hf = 20
        self.vf = 1800
        self.Q_dot_max_allow = 1200  # Kw/m2
        self.n_max_allow = 4  # g0
        self.q_max_allow = 200  # Kpa

    def get_x_dot(self, x, alpha, tht, p=0):
        # 对当前状态求导，得到状态导数
        # 输入当前状态、攻角(度)、倾侧角(度)、推力（沿机体坐标系x），输出状态导数
        # 状态提取[半径m，维度，经度，速度m/s，弹道倾角(弧度)，弹道偏角(弧度)]
        r, gamma, phi, v, theta, chi = x

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
        # range_dot = v * math.cos(theta) / r
        Q_dot = self.C1 / math.sqrt(self.Rd) * (rho / self.rho0) ** 0.5 * (v / self.Vc) ** 3.15
        info = {"Q_dot": Q_dot, "ny": ny, "q": q / 1000, "tht": tht, "alpha": alpha}

        x_dot = np.array([r_dot, gama_dot, phi_dot, v_dot, theta_dot, chi_dot])
        return x_dot, info

    def v2alpha(self, v):
        # 输入速度m/s,输出攻角(度),这是CAV本身的设计决定的,不需要改变
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
        # cav-l数据，最大升阻比为2.4
        if self.low == True:
            # 低升阻比数据
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
            c1, c2 = 0, 0
        return cl, cd


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
        self.delta_t = 0.01  # 积分步长
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
        self.x0 = np.hstack((self.r0 * 1000, self.gama0, self.phi0, self.v0, self.theta0, self.chi0))

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

    def step(self, action):
        # 假设倾侧角是action度
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


if __name__ == '__main__':
    cav = AircraftEnv()
    state_now = cav.reset()
    state_record = state_now.copy()
    for i in range(10000):
        action = np.random.rand(1) * 180
        state_now, reward, done, info = cav.step(action)
        state_record = np.vstack((state_record, state_now.copy()))  # 垂直添加
plt.plot(state_record[:, 3])
plt.show()
