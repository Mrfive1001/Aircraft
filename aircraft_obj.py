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

    def get_state_dot(self, x, alpha, tht, p=0):
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

        state_dot = np.array([r_dot, gama_dot, v_dot, theta_dot, chi_dot])
        return state_dot, info

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

    def alphama2clcd(self, alpha, ma, low=True):
        # 攻角(度)马赫数转换为升力系数、阻力系数
        # cav-l数据，最大升阻比为2.4
        if low == True:
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
            pass
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
