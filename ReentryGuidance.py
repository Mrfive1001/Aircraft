import numpy as np
import scipy.io as sio
import scipy.interpolate as interploate
import math

class ReentryGuidance(object):

    def __init__(self):

        # 仿真数据
        self.timestep = 1
        self.s_dim = 8
        self.a_dim = 1
        self.a_bound = np.array([0.5,0.5])  # 0到1


        # 飞行器总体参数
        self.m0 = 907  #kg
        self.S = 0.4837  #m2
        self.R0 = 6378.135  #Km
        self.g0 = 9.81  #m/s^2
        self.Rd = 0.1
        self.C1 = 11030
        self.Vc = (self.g0*self.R0*1000)**0.5
        self.Tc = (self.R0*1000/self.g0)**0.5
        self.beta = 7200
        self.rho0 =1.225

        # 飞行器初始位置信息
        self.h0 = 90  #Km
        self.r0 = self.R0 + self.h0  #Km
        self.v0 = 7000   #m/s
        self.theta0 = -2 / 180 * math.pi  #弧度
        self.chi0 = 55 / 180 * math.pi  #弧度
        self.gama0 = 140 / 180 * math.pi  #弧度
        self.phi0 = 0 / 180 * math.pi #弧度
        self.range0 = 0  #Km
        self.time=0
        self.X0_2 = np.hstack((self.r0*1000, self.range0, self.v0, self.theta0))
        self.X0_3=np.hstack((self.r0*1000, self.gama0, self.phi0, self.v0, self.theta0, self.chi0, self.range0, self.time))

        # 目标位置信息
        self.gamaT0 = 215 / 180 * math.pi  #弧度
        self.phiT0 = 25 / 180 * math.pi  #弧度
        self.vT0 = 0  #弧度
        self.chiT0 = 0  #弧度
        self.range_need = self.CalRange(self.gama0, self.phi0, self.gamaT0, self.phiT0)  #Km
        self.angle_need = self.CalAngle(self.gama0, self.phi0, self.gamaT0, self.phiT0)

        # 约束条件
        self.hf = 16
        self.vf = 850
        self.Q_dot_max_allow = 800   #Kw/m2
        self.n_max_allow = 6    # g0
        self.q_max_allow = 100  #Kpa
        self.w_Q_dot = 0.1 #1
        self.w_ny = 0.1 #10
        self.w_q = 0.1 #1
        self.time_need = 1900
        self.delta_range = self.range_need / self.time_need

        # 求解好的走廊
        self.Corridor = sio.loadmat('./NewCorridor.mat')
        self.vv_HV = self.Corridor['vv']
        self.H_up_HV = self.Corridor['HV_up']
        self.H_down_HV = self.Corridor['HV_down']
        self.tht_up = self.Corridor['tht_max']
        self.tht_down = - self.tht_up #self.Corridor['tht_min']#
        self.Range_profile = sio.loadmat('./range_profile.mat')
        self.Weight_profile = self.Range_profile['weight']
        self.w2range_profile = self.Range_profile['range']

    def reset_stable(self):
        return self.X0_3

    def reset_random(self):
        self.gama0 = np.random.normal(140 , 2, 1) / 180 * math.pi
        self.phi0 = np.random.normal(0 , 2 , 1) / 180 * math.pi
        self.range_need = self.CalRange(self.gama0, self.phi0, self.gamaT0, self.phiT0)
        self.X0_3=np.hstack((self.r0*1000, self.gama0, self.phi0, self.v0, self.theta0, self.chi0, self.range0, self.time))
        return self.X0_3


    def step(self, state, weight, var, RLmethod):


        reward_sum = 0
        reward = 100
        done = False

        state_init = state

        for kk in range(1):

            k1, info = self.move_3D(state, weight)
            state_next = state + k1

            Q_dot_exceed = np.max(np.array([0, info[0]-self.Q_dot_max_allow]))
            ny_exceed = np.max(np.array([0, info[1]-self.n_max_allow]))
            q_exceed = np.max(np.array([0, info[2]-self.q_max_allow]))
            if max(Q_dot_exceed, ny_exceed, q_exceed) >0:
                print('exceed:', max(Q_dot_exceed, ny_exceed, q_exceed))

            reward_constraint = self.w_Q_dot * Q_dot_exceed + self.w_ny * ny_exceed + self.w_q * q_exceed
            angle = self.CalAngle(state_next[1], state_next[2], self.gamaT0, self.phiT0)
            rang = self.CalRange(state_next[1], state_next[2], self.gamaT0, self.phiT0)

            reward_sum -= reward_constraint


            #if state_next[7] <= self.time_need / 2 or state_next[7] > self.time_need:
            #    reward_sum += 0.01
            #else:
            #   reward_sum -= 0.01


            if kk==0:
                state_return=state_next
                info_return=info


            if state_next[3] < self.vf:

                done = True
                break
            else:
                done = False
                state = state_next

                state_feature = self.State_Feature_extraction(state)
                action = RLmethod.choose_action_eval(state_feature)
                weight = np.clip(action, 0, 1)


        if done:# and kk <= 2:
            range_error = self.CalRange(state_next[1],state_next[2],self.gamaT0,self.phiT0)
            reward = reward_sum - 10 * ((state_return[1] * 57.3 - self.gamaT0 * 57.3)**2 + (state_return[2] * 57.3 - self.phiT0 * 57.3)**2)**0.5
            reward = reward_sum - 0.1 * abs(self.time_need - state_return[7])
            #reward = reward_sum - 500 * (range_error / 1000) ** 2 - 50*(abs(state_next[7] - self.time_need - 1) / 100)**2 + 10
            #print('rrrrr_end:  ', reward)

        else:
            done = False
            range_error_old = self.CalRange(state_init[1], state_init[2], self.gamaT0, self.phiT0)
            range_error = self.CalRange(state_next[1], state_next[2], self.gamaT0, self.phiT0)
            reward = reward_sum

            los = (self.CalAngle(state_return[1], state_return[2], self.gamaT0, self.phiT0) - state_return[5]) * 180 / math.pi
            if los >= 180:
                los -= 180
            elif los <= -180:
                los += 180

        #print('range_to_go:    ',range_error,'time:     ',state_return[7])
        #return state, reward, done, info
        return state_return, reward, done, info_return


    def V2Alpha(self, v):

        vv = 3400
        alpha_max = 35
        if v < vv:
            alpha = alpha_max-0.45*(v/340-10) ** 2
        else:
            alpha = alpha_max

        return alpha

    def V2Tht(self, v, weight):

        if v > self.vv_HV.max():
            v = self.vv_HV.max()
        elif v < self.vv_HV.min():
            v = self.vv_HV.min()

        tht_design = (1 - weight) * self.tht_up[0] + weight * self.tht_down[0]
        #tht_design = weight * self.tht_up[0]
        # "nearest","zero"为阶梯插值
        # slinear 线性插值
        # "quadratic","cubic" 为2阶、3阶B样条曲线插值
        f = interploate.interp1d(self.vv_HV[0], tht_design, kind='quadratic')
        return f(v)
        #return tht

    def H2g(self, h):

        # h的单位为m
        g = self.g0 * (self.R0 * 1000 )**2 / (self.R0 * 1000 + h)**2
        return g

    def H2rho(self, h):

        # h的单位为m
        rho = self.rho0 * math.exp(-h / self.beta)
        return rho

    def CalRange(self,lamda_a,phi_a,lamda_c,phi_c):
        #剩余射程
        ad = 2 * math.cos(phi_a) * math.sin((lamda_c - lamda_a) / 2)
        ec = 2*math.cos(phi_c)*math.sin((lamda_c-lamda_a)/2)
        cd = 2*math.sin((phi_c-phi_a)/2)
        ac = (ad*ec+cd**2)**0.5
        range_need=2*math.asin(ac/2)*self.R0

        return range_need
    def CalAngle(self, lamda_a, phi_a, lamda_c, phi_c):
        #当前位置与目标的视线角
        angle = math.atan(math.sin(lamda_c - lamda_a) / (math.cos(phi_a) * math.tan(phi_c) - math.sin(phi_a) * math.cos(lamda_c - lamda_a)))
        angle = angle * (180 / math.pi)
        if angle < 0 and lamda_c > lamda_a:
            angle = angle + 180
        if angle > 0 and lamda_c < lamda_a:
            angle = angle - 180
        angle = angle / (180 / math.pi)
        return angle

    def MK_move_equation2(self, state, weight):  #Mangge Kuta 45

        # state decompose
        r = state[0]
        h = r - self.R0 * 1000
        range = state[1]
        v = state[2]
        theta = state[3]

        # environment calculation
        g = self.H2g(h)
        rho = self.H2rho(h)

        # control
        alpha = self.V2Alpha(v)
        tht = self.V2Tht(v, weight)
        Cl, Cd = self.AlphaMa2ClCd(alpha, v/340)
        q = 0.5 * rho * v**2
        L = q * Cl * self.S
        D = q * Cd * self.S


        # 运动方程
        r_dot = v * math.sin(theta)
        range_dot = v * math.cos(theta)/r
        v_dot = -D / self.m0 - g * math.sin(theta)
        theta_dot = 1/v * (L * math.cos(tht/180 * math.pi) / self.m0 - (g - v**2/r) * math.cos(theta))

        state_dot = np.hstack((r_dot, range_dot, v_dot, theta_dot))

        # path constraint
        Q_dot = self.C1 / math.sqrt(self.Rd) * (rho / self.rho0) ** 0.5 * (v / self.Vc) ** 3.15
        ny = math.sqrt((L / self.m0 / self.g0) ** 2 + (L / self.m0 / self.g0) ** 2)
        q= q
        info = {"Q_dot": Q_dot, "ny": ny, "q": q/1000, "tht": tht}

        return state_dot, info

    def move_3D(self,state,weight):
        # 顺序r gama phi v theta chi range time
        r = state[0]
        h = r - self.R0 * 1000
        gama=state[1]
        phi=state[2]
        v = state[3]
        theta = state[4]
        chi=state[5]

        # environment calculation
        g = self.H2g(h)
        rho = self.H2rho(h)

        # control
        alpha = self.V2Alpha(v)
        tht = self.V2Tht(v, weight)


        Cl, Cd = self.AlphaMa2ClCd(alpha, v / 340)
        q = 0.5 * rho * v ** 2
        L = q * Cl * self.S
        D = q * Cd * self.S

        # 运动方程
        #顺序r0 gama0 phi0 v0 theta0 chi0 range time
        r_dot = v * math.sin(theta)
        gama_dot = v * math.cos(theta) * math.sin(chi) / r / math.cos(phi)
        phi_dot = v * math.cos(theta) * math.cos(chi) / r
        v_dot = - D / self.m0 - g * math.sin(theta)
        theta_dot = 1 / v * (L * math.cos(tht / 180 * math.pi) / self.m0 - (g - v ** 2 / r) * math.cos(theta))
        chi_dot = 1 / v * (L * math.sin(tht / 180 * math.pi) / math.cos(theta) / self.m0 + (v**2/r)*math.cos(theta)*math.sin(chi)*math.tan(phi))
        range_dot = v * math.cos(theta) / r
        time_dot=1

        #print(r_dot, v_dot,'\n')
        state_dot = np.hstack((r_dot, gama_dot, phi_dot, v_dot, theta_dot, chi_dot, range_dot, time_dot))

        #path constraint
        Q_dot = self.C1 / math.sqrt(self.Rd) * (rho / self.rho0) ** 0.5 * (v / self.Vc) ** 3.15
        ny = math.sqrt((L / self.m0 / self.g0) ** 2 + (D / self.m0 / self.g0) ** 2)
        q = q
        info = np.hstack((Q_dot, ny, q/1000, tht))
        return state_dot, info

    def AlphaMa2ClCd(self, alpha, Ma):

        CL=-0.0196-0.02065*Ma+0.002347*Ma**2-(8.486e-05)*Ma**3+(9.987e-07)*Ma**4+\
            0.05097*alpha-0.0009715*alpha*Ma-(3.141e-06)*alpha*Ma**2+(4.148e-07)*alpha*Ma**3+\
            0.0004383*alpha**2-(9.948e-06)*alpha**2*Ma+(2.897e-07)*alpha**2*Ma**2

        CD=0.4796-0.07427*Ma+0.003595*Ma**2-(7.542e-05)*Ma**3+(1.029e-06)*Ma**4+\
           (-0.04037)*alpha+0.005605*alpha*Ma-0.0001142*alpha*Ma**2+(-2.286e-06)*alpha*Ma**3+\
           0.002492*alpha**2-(0.0002678)*alpha**2*Ma+(8.114e-06)*alpha**2*Ma**2

        return CL, CD

    def State_Feature_extraction(self, state):
        # 顺序r0 gama0 phi0 v0 theta0 chi0 range time
        r = state[0]
        h = r - self.R0 * 1000
        gama=state[1]
        phi=state[2]
        v = state[3]
        theta = state[4]
        chi=state[5]
        range = state[6]
        time=state[7]

        h_feature = (h / 1000 - (self.h0 + self.hf) / 2) / (self.h0 - self.hf)
        v_feature = (v - (self.v0 + self.vf) / 2) / (self.v0 - self.vf)
        gama_feature = (gama - (self.gamaT0 + self.gama0) / 2) / (self.gamaT0 - self.gama0)
        phi_feature = (phi - (self.phiT0 + self.phi0) / 2) / (self.phiT0 - self.phi0)
        theta_feature = theta / 0.3
        chi_feature = chi / 0.3

        range_togo = self.CalRange(gama, phi, self.gamaT0, self.phiT0)
        range_feature = range_togo / self.range_need

        time_feature = (self.time_need - time) / self.time_need
        angle_need = self.CalAngle(gama,phi,self.gamaT0,self.phiT0)
        los_feature = angle_need - chi

        if los_feature <= - math.pi:
            los_feature += math.pi
        if los_feature >= math.pi:
            los_feature -= math.pi



        # 顺序r0 gama0 phi0 v0 theta0 chi0 range time
        #state_feature = np.hstack((h_feature, v_feature, theta_feature, chi_feature, range_feature, time_feature, los_feature))
        state_feature = np.hstack((h_feature, gama_feature, phi_feature, v_feature, theta_feature, chi_feature, range_feature, time_feature))

        return state_feature
