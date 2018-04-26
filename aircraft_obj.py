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

