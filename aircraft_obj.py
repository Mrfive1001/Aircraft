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
        d = q * cd * self.S  # 阻力
        l = q * cl * self.S  # 升力
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
        # 给出两点的经纬度计算射程 射程单位km
        ad = 2 * math.cos(phi0) * math.sin((gamma1 - gamma0) / 2)
        ec = 2 * math.cos(phi1) * math.sin((gamma1 - gamma0) / 2)
        cd = 2 * math.sin((phi1 - phi0) / 2)
        ac = (ad * ec + cd ** 2) ** 0.5
        range = 2 * math.asin(ac / 2) * self.R0
        return range

    def phigamma2angle(self, gamma0, phi0, gamma1, phi1):
        # 当前位置与目标的视线角
        # 输入弧度，输出弧度，表示和目标连线与正北方向的夹角
        angle = math.atan(math.sin(gamma1 - gamma0) / (math.cos(phi0) * math.tan(phi1) -
                                                       math.sin(phi0) * math.cos(gamma1 - gamma0)))
        angle = angle * (180 / math.pi)
        if angle < 0 and gamma1 > gamma0:
            angle = angle + 180
        if angle > 0 and gamma1 < gamma0:
            angle = angle - 180
        angle = angle / (180 / math.pi)
        return angle


class AircraftEnv(CAV):
    """
    编写飞行器Markov模型
    分为几个部分
    x 表示内部飞行器所有信息
    state 外部所需要的信息
    """

    def __init__(self, random=False, low=True, start_gammaphi=[160, 5], end_gammadphi=[225, 25]):
        CAV.__init__(self, low)
        self.x = None  # 内部循环状态变量
        self.state = None  # 深度强化学习中或者与外界交互变量，长度小于x
        self.t = 0
        self.delta_t = 0.5  # 积分步长
        self.random = random

        # 飞行器初始位置信息
        self.h0 = 100  # Km
        self.r0 = self.R0 + self.h0  # Km
        self.v0 = 7200  # m/s
        self.theta0 = -2 / 180 * math.pi  # 弧度  弹道倾角
        self.chi0 = 55 / 180 * math.pi  # 弧度    弹道偏角
        self.gama0 = start_gammaphi[0] / 180 * math.pi  # 弧度  经度
        self.phi0 = start_gammaphi[1] / 180 * math.pi  # 弧度     维度
        self.range0 = 0  # Km   射程
        self.Q0 = 0
        self.Osci0 = 0
        self.x0 = np.hstack((self.r0 * 1000, self.gama0, self.phi0, self.v0, self.theta0, self.chi0, self.range0))

        # 飞行器终点位置 终端位置高度速度射程
        self.hf = 20  # km
        self.rf = self.R0 + self.hf  # km
        self.vf = 1800  # m/s
        self.gamaf = end_gammadphi[0] / 180 * math.pi  # 弧度
        self.phif = end_gammadphi[1] / 180 * math.pi  # 弧度
        self.rangef = self.phigamma2range(self.gama0, self.phi0, self.gamaf, self.phif)  # 射程km 与 range*self.R0差不多
        self.anglef = self.phigamma2angle(self.gama0, self.phi0, self.gamaf, self.phif)
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
        # 暂存变量
        self.hv_temp = None
        # 环境维度
        self.s_dim = len(self.state)
        self.a_dim = None

        self.errors = 0

    def _x2state(self, x):
        # 将x缩减为state
        return x.copy()

    def render(self):
        pass

    def reset(self):
        # 重置函数
        self.t = 0
        if self.random:
            pass
        else:
            self.x = self.x0.copy()
        self.state = self._x2state(self.x)
        self.errors = 0
        return self.state.copy()

    def hv_w(self, w):
        # 输入一个w值自动规划出一条弹道[0~1] w越大，离上界越近
        # 返回弹道记录
        state_now = self.reset()
        state_record = state_now.copy()
        cons = 22
        control = [cons]
        v2h_down = interp1d(self.vv, self.h_down, kind='quadratic')
        v2h_up = interp1d(self.vv, self.h_up, kind='quadratic')
        h_cmds = []
        info = {}
        store = []
        v_init = 7000  # 初始下滑段的速度界限
        while True:
            v = self.x[3]
            h_cmd = (1 - w) * v2h_down(min(v, self.v0)) + w * v2h_up(min(v, self.v0))  # m
            h = state_now[0] - self.R0 * 1000  # 高度m
            # if math.fabs(h-h_ref)<100:
            if v < 5500:
                temp = np.array([w, v, state_now[-1]])
                if store == []:
                    store = temp.copy()
                else:
                    store = np.vstack((store, temp.copy()))
            if v > v_init:
                tht = cons / 57.3
            else:
                tht = self.h2tht(h_cmd, h_cmds)
            h_cmds.append(h_cmd)
            control.append(tht * 57.3)
            state_now, reward, done, info = self.step(tht * 57.3)

            state_record = np.vstack((state_record, state_now.copy()))  # 垂直添加
            if done:
                h_cmds.append(h_cmd)
                break
        store[:, -1] = (np.max(store[:, -1]) - store[:, -1]) * self.R0  # 预计射程km
        info['control'] = np.array(control)
        info['h_cmds'] = np.array(h_cmds)
        info['store'] = store
        return state_record.copy(), info

    def h2tht(self, h_cmd, h_cmds, thts=None):
        # 利用倾侧角跟踪目标高度
        # 追踪参数
        lambda_h = 0.04
        h_cmd_dot = (h_cmd - h_cmds[-1]) / self.delta_t
        h_cmd_dot2 = ((h_cmd - h_cmds[-1]) - (h_cmds[-1] - h_cmds[-2])) / (self.delta_t) ** 2
        # 气动方面计算
        v = self.x[3]
        r = self.x[0]
        h = r - self.R0 * 1000  # 高度
        theta = self.x[4]
        g = self.h2g(h)
        rho = self.h2rho(h)
        alpha = self.v2alpha(v)
        cl, cd = self.alphama2clcd(alpha, v / 340)
        q = 0.5 * rho * v ** 2  # 动压
        l = q * cl * self.S  # 升力
        d = q * cd * self.S  # 阻力
        # 计算各导数
        v_dot = - d / self.m0 - g * math.sin(theta)
        h_dot = v * math.sin(theta)

        delta_h = h - h_cmd
        delta_h_dot = h_dot - h_cmd_dot

        h_dot2 = 0 - 2 * delta_h_dot * lambda_h - lambda_h ** 2 * delta_h

        cos_tht = ((h_dot2 - v_dot * math.sin(theta)) / math.cos(theta) +
                   (g - v ** 2 / r) * math.cos(theta)) * self.m0 / l
        cos_tht = cos_tht if (0 <= cos_tht <= 1) else (1 if cos_tht > 1 else 0)
        tht = math.acos(cos_tht)  # 弧度
        if thts:
            por = 0.9
            tht = por * abs(thts[-1]) + (1 - por) * tht
        return tht

    def hv2tht(self, h, v):
        # h 单位m，v单位m/s
        # 计算平衡滑翔角
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
        n_maxs = [4, 3]
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
        h_down1 = v.copy()  # 下界
        h_down2 = v.copy()  # 下界
        h_down_change = v.copy()  # 人为设定的末端下界
        # h的上界值，h太高会掉下来
        h_qegc1 = v.copy()  # 平衡滑翔条件
        h_qegc2 = v.copy()  # 平衡滑翔条件
        h_qegc_change = v.copy()  # 人为设定的末端上界

        def f(h, vv, tht):
            # 计算对应速度和高度，判断倾侧角是否为平衡条件 h单位m v单位m/s
            g = self.h2g(h)
            rho = self.h2rho(h)
            r = self.R0 * 1000 + h
            q = 0.5 * rho * vv ** 2  # 动压
            l = q * cl * self.S  # 升力
            return l * math.cos(tht) / self.m0 + (vv ** 2 / r) - g

        def f_up_down(v, mode, h2v, self=self):
            # mode==0指上边界
            # 计算如果是上边界在当前速度高度下，按照最大倾侧角飞行，与目标点的距离
            h = h2v(v)
            x = self.x0.copy()
            x[0], x[3] = h + self.R0 * 1000, v
            action = 90 if mode == 0 else 0
            vs = []
            hs = []
            while True:
                v = x[3]
                alpha = self.v2alpha(v)
                # 简单积分，积分时间越短越精确
                x_dot, info = self.get_x_dot(x, alpha, action)
                x += x_dot * 0.01
                h = x[0] - self.R0 * 1000
                vs.append(v)
                hs.append(h)
                # 判断是否截止
                h_diff = h - self.hf * 1000
                if x[3] <= self.vf:
                    ceq = math.fabs(h_diff / 10) + math.fabs(x[3] - self.vf)
                    break
                elif (mode == 0 and h_diff <= 0) or (mode == 1 and h_diff >= 0):
                    ceq = math.fabs(x[3] - self.vf)
                    break
            self.hv_temp = dict(vs=np.array(vs), hs=np.array(hs))
            return ceq

        for i in range(len(v)):
            # 计算不同约束下的上下边界
            for tht, n_max, Q_max, q_max, h_nmax, h_Qmax, h_qmax, h_qegc, h_down in zip(
                    thts, n_maxs, Q_maxs, q_maxs, [h_nmax1, h_nmax2], [h_Qmax1, h_Qmax2], [h_qmax1, h_qmax2],
                    [h_qegc1, h_qegc2], [h_down1, h_down2]):
                h_Qmax[i] = 2 * self.beta * math.log((self.C1 * (v[i] / self.Vc) ** 3.15) /
                                                     (Q_max * (self.Rd ** 0.5)))
                alpha = self.v2alpha(v[i])
                cl, cd = self.alphama2clcd(alpha, v[i] / 340)
                h_nmax[i] = 1 * self.beta * math.log((self.rho0 * v[i] ** 2 * self.S * (cl ** 2 + cd ** 2) ** 0.5) /
                                                     (2 * n_max * self.m0 * self.g0))
                h_qmax[i] = 1 * self.beta * math.log((self.rho0 * v[i] ** 2) / (2 * q_max * 1000))

                h_down[i] = max(h_Qmax[i], h_nmax[i], h_qmax[i])
                # 计算平衡滑翔的对应的高度
                res = root(f, 0, (v[i], tht))
                h_qegc[i] = res.x
        # 将末端走廊规划到一点，减少一个末端约束
        v2h_down = interp1d(v, h_down1, kind='quadratic')
        v2h_up = interp1d(v, h_qegc1, kind='quadratic')
        # 找到上边界的临界点
        res = root(f_up_down, 2000, (0, v2h_up))
        v_up_change = res.x[0]
        f_up_down(v_up_change, 0, v2h_up)
        vs_up = self.hv_temp['vs'].copy()
        hs_up = self.hv_temp['hs'].copy()
        v2h_up = interp1d(vs_up, hs_up, kind='quadratic')
        # 找出下边界零度倾侧角的临界点
        res = root(f_up_down, 2000, (1, v2h_down))
        v_down_change = res.x[0]
        f_up_down(v_down_change, 1, v2h_down)
        vs_down = self.hv_temp['vs'].copy()
        hs_down = self.hv_temp['hs'].copy()
        v2h_down = interp1d(vs_down, hs_down, kind='quadratic')
        # 将其进行插值得到最终的走廊
        for i in range(len(v)):
            if v[i] >= v_down_change:
                h_down_change[i] = h_down1[i]
            else:
                h_down_change[i] = v2h_down(min(max(v[i], min(vs_down)), max(vs_down)))
            if v[i] >= v_up_change:
                h_qegc_change[i] = h_qegc1[i]
            else:
                h_qegc_change[i] = v2h_up(min(max(v[i], min(vs_up)), max(vs_up)))

        np.savez('corrior_orginal', vv=v, h_down=h_down, h_up=h_qegc)
        np.savez('corrior', vv=v, h_down=h_down_change, h_up=h_qegc_change)

        # 画图
        fig_HV = plt.figure()
        ax = fig_HV.add_subplot(111)
        plt.plot(v, h_qegc1 / 1000, v, h_qegc2 / 1000, '--', v, h_qmax1 / 1000, v, h_qmax2 / 1000, '-.', v,
                 h_Qmax1 / 1000,
                 v, h_Qmax2 / 1000, ':', v, h_nmax1 / 1000, v, h_nmax2 / 1000, ':')
        plt.annotate('$Q_{max} = %s Kw/m^2$' % (Q_maxs[0]), (2503, -3.7), (2892, -7),  # 文字，图像中的坐标，文字的坐标
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$Q_{max} = %s Kw/m^2$' % (Q_maxs[1]), (2293, -9), (2792, -14.3),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$q_{max} = %s KPa$' % (q_maxs[0]), (6290, 34.7), (6700, 24),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$q_{max} = %s KPa$' % (q_maxs[1]), (6036, 28.9), (6610, 20),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$n_{max} = %s $' % (n_maxs[1]), (2373.3, 18.43), (2318, 4),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$n_{max} = %s $' % (n_maxs[0]), (1834.4, 12.9), (2004, 0),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$tht = %s^\circ$' % (thts[0]), (4519, 43.4), (4185, 55),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.annotate('$tht = %.2s^\circ $' % (thts[1] * 57.3), (3850, 36), (3900, 50),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        ax.set_xlabel(r'v(m/s)')
        ax.set_ylabel(r'h(km)')
        ax.grid()
        plt.savefig(os.path.join('Figures', 'HV走廊敏感性分析.png'))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(v, h_qegc_change / 1000, v, h_down_change / 1000, v, h_down1 / 1000, v, h_qegc1 / 1000)
        ax.set_xlabel(r'v(m/s)')
        ax.set_ylabel(r'h(km)')
        plt.legend(['h_qegc_change', 'h_down_change', 'h_down', 'h_qegc'])
        plt.grid()
        plt.savefig(os.path.join('Figures', 'HV走廊有效性.png'))

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

    def calculate_range(self):
        return self.phigamma2range(self.state[1], self.state[2], self.gamaf, self.phif)

    def calculate_angle(self):
        # 视线角差
        return self.phigamma2angle(self.state[1], self.state[2], self.gamaf, self.phif) - self.state[-2]

    def plot(self, data, hcmds=None, ax=None):
        # 画出轨迹的图，输入数据每行代表某一个时刻的状态量
        # 将规划的hv也画出来
        t = np.arange(0, len(data), 1) * self.delta_t  # 时间
        if ax:
            ax.plot(self.vv, self.h_up / 1000, self.vv, self.h_down / 1000)
            ax.plot(data[:, 3], data[:, 0] / 1000 - self.R0)
            if hcmds is not None:
                ax.plot(data[:, 3], np.array(hcmds) / 1000)
            ax.grid()
        else:
            plt.plot(self.vv, self.h_up / 1000, self.vv, self.h_down / 1000)
            plt.plot(data[:, 3], data[:, 0] / 1000 - self.R0)
            if hcmds is not None:
                plt.plot(data[:, 3], np.array(hcmds) / 1000)
            plt.grid()


if __name__ == '__main__':
    # cav = AircraftEnv()
    # cav.h_v()  # 进行HV走廊敏感性分析和有效性分析
    cav = AircraftEnv()
    num = 1
    for i in range(num + 1):
        s, info = cav.hv_w(i / num)
        cav.plot(s)
    plt.plot(s[:, 3], info['h_cmds'] / 1000)
    plt.show()
