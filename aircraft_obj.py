import numpy as np
import math


class AircraftEnv:
    """
    编写飞行器Markov模型
    分为几个部分
    x 表示内部飞行器所有信息
    state 外部所需要的信息
    """

    def __init__(self):
        pass

    def render(self):
        pass

    def reset(self):
        pass

    def step(self):
        pass


class CAV:
    """
    CAV的飞行参数与气动模型
    """

    def __init__(self):
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

        # 约束条件
        self.hf = 20
        self.vf = 1800
        self.Q_dot_max_allow = 1200  # Kw/m2
        self.n_max_allow = 4  # g0
        self.q_max_allow = 200  # Kpa
        self.w_Q_dot = 1  # 1
        self.w_ny = 1  # 10
        self.w_q = 0.1  # 1

    def v2alpha(self, v):
        # 输入速度m/s,输出攻角,这是CAV本身的设计决定的,不需要高边
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
        # 攻角马赫数转换为升力系数、阻力系数
        # cav-l数据，最大升阻比为2.4
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
        return cl, cd


if __name__ == '__main__':
    cav = CAV()
    mas = np.array([3.5, 5, 8, 15, 20, 23])
    alphas = np.array([10, 15, 20])
    cls = []
    cds = []
    for alpha in alphas:
        temp_cls, temp_cds = [], []
        for ma in mas:
            temp_cl, temp_cd = cav.alphama2clcd(alpha, ma)
            temp_cls.append(temp_cl)
            temp_cds.append(temp_cd)
        cls.append(temp_cls.copy())
        cds.append(temp_cds.copy())
