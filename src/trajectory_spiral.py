"""
ref : Refinements to the Q-law for low-thrust orbit transfers
単位はrad, km, s, kg
f_vec = [f_r, f_theta, f_h]
"""

from libs.lib import Values, TrajectoryCalculator, Planet
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from datetime import datetime
import os
import json

#Values
myval = Values()
#Planet
earth = Planet("Earth")
mars = Planet("Mars")
mars_calclator = TrajectoryCalculator("Mars")

mu = mars.mu

# params---------------------------------------------------------------------
rh = 10 * mars.radius
r_max = 5.78 * 10**5 # 火星の重力影響圏
e0 = 1 - 1e-5
m0 = 200
t_end = 100* 24 * 60 * 60
eta = 0

thrust = 45 * 1e-3 * 1e-3 # はやぶさ 10 mN = 10 * 1e-3 * 1e-3 kg km/s^2
I_sp = 3000
# thrust = 45 * 1e-3 * 1e-3 # Masmi 45 mN
# I_sp = 1733
# thrust = 89 * 1e-3 * 1e-3 # SPT100
# I_sp = 1562

w_p = 1
w_oe = np.array([[5, 5, 10**(-2), 10**(-2), 10**(-2)]]).T
r_p_min = mars.radius*1.05
# ---------------------------------------------------------------------------
g = 9.8 * 1e-3 # km/s^2
# 初期条件
h0 = (2 * mu * rh)**0.5 
p0 = h0**2 / mu
a0 = p0 / (1 - e0**2)
theta0 = - np.arccos(1 / e0 * (p0 / r_max - 1)) # スタート点が火星の重力影響圏

oe_0 = np.array([[a0, e0, 80, 0.01, 1]]).T # [a, e, i, Omega, omega]
# oe_0 = np.array([[7829, 0.4018, 30.50, 2.255, 16.75]]).T
oe_t = np.array([[7000, 0.01, 80, 0.01, 1]]).T 
# oe_0 = np.array([[42000, 0.01, 0.050, 0, 0]]).T 

now = datetime.now()
folder_name = "outputs/runs"
file_name = f"{folder_name}/{now.strftime('%Y%m%d%H%M%S')}.json"
os.makedirs(folder_name, exist_ok=True)

def main():
    spiral = Spiral(w_p, w_oe, r_p_min, oe_0, oe_t, m0, thrust, t_end, eta)
    spiral.simulate()
    
    r_his = spiral.r_his
    m_his = spiral.m_his
    dV_his = spiral.dV_his
    non_zero_index = np.where(m_his != 0)
    m_his = m_his[non_zero_index]
    r_his = r_his[:,non_zero_index][:,0,:]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_his[0,:], r_his[1,:], r_his[2,:])
    plt.show()
    plt.plot(spiral.t_his[non_zero_index], m_his)
    plt.show()
    dV_tot = np.sum(spiral.dV_his)
    print(dV_tot)

    with open(file_name, "w") as f:
        json.dump(r_his.tolist(), f, indent=" ")
        json.dump(dV_his.tolist(), f, indent=" ")

def runge_kutta_1step(func, y, dt):
    """ルンゲクッタで1ステップ進める

    Args:
        func (function(y)):dydtを返す関数
        y (ndarray):初期値(n*1)
        t (double): time
        dt (double): time step

    Returns:
        ndarray: dt後のy(n*1)
    """
    k0 = func(y) * dt
    k1 = func(y + k0 / 2.0) * dt
    k2 = func(y + k1 / 2.0) * dt
    k3 = func(y + k2) * dt
    y_next = y + (k0 + 2.0 * k1 + 2.0 * k2 + k3) / 6.0
    return y_next

class Spiral():
    def __init__(self, w_p, w_oe, r_p_min, oe_0, oe_t, m0, thrust, t_end, eta, dt_initial=1, dt_nominal=50, dt_skip=500):
        """init

        Args:
            w_p (float): weight for P
            w_oe (ndarray): weights for oe(1*5)
            r_p_min (float): min of perigee distance
            oe_0 (ndarray): initial oes(1*5) [a, e, i, Omega, omega]
            oe_t (ndarray): taraget oes(1*5) [a, e, i, Omega, omega]
            m0 (float): initial mass[kg]
            thrust (float): thrust[N]
            t_end (float): time at end of transition
            eta (float): cut off efficiency
            dt (int, optional): time step. Defaults to 10.
        """
        self.dt_initial = dt_initial #fix medtを変えるロジックについて現在の実装だと遠くから近づくのにだけ対応していて、逆は非対応
        self.dt_nominal = dt_nominal
        self.dt_skip = dt_skip
        self.w_p = w_p
        self.w_oe = w_oe
        self.r_p_min = r_p_min
        self.oe0 = oe_0
        self.oe_t = oe_t
        self.m0 = m0
        self.thrust = thrust
        self.t_end = t_end
        self.eta = eta
        # fix me
        # dtを変えるロジックについて現在の実装だと遠くから近づくのにだけ対応していて、逆は非対応
        self.dt = dt_initial # 計算量を減らすために途中で変更するので注意
        self.a_threshold = 5 * 10**6
        self.num = int(self.t_end / self.dt)
        
        self.oe_his = np.zeros((5,self.num))
        self.theta_his = np.zeros(self.num)
        self.r_his = np.zeros((3,self.num))
        self.t_his = np.zeros(self.num)
        self.dV_his = np.zeros(self.num)
        self.m_his = np.zeros(self.num)
    
    def simulate(self):
        t = 0
        i = 0
        m = self.m0
        oe = self.oe0
        theta = theta0
        self.thrust_enabled = True
        fig, ax = plt.subplots()

        while(t < self.t_end):
            f_norm = self.thrust / m
            f_vec = self.calc_input(oe, theta, f_norm)

            oe_temp, theta, m = self.update(oe, theta, m, f_vec, t)
            if(oe_temp[1] >= 1):
                continue
            oe = oe_temp

            t += self.dt
            i += 1
            self.oe_his[:, i] = oe[:,0]
            self.theta_his[i] = theta
            self.t_his[i] = t
            self.m_his[i] = m
            if (self.thrust_enabled):
                self.dV_his[i] = f_norm * self.dt

            P_hat_vec, Q_hat_vec, _ = mars_calclator.calc_PQW_from_orbital_elems(oe[0,0], oe[1,0], np.rad2deg(oe[2,0]), np.rad2deg(oe[4,0]), np.rad2deg(oe[3,0]), 0)
            r,_ = mars_calclator.calc_rv_from_nu(np.rad2deg(theta), oe[0,0], oe[1,0], P_hat_vec, Q_hat_vec)
            self.r_his[:, i] = r[:,0]

            dif = np.dot(self.w_oe.T, self.distance(oe))
            if (i % 50 == 0):
                print('time[hour]: ', t / 60 / 60, 'percent: ', t / self.t_end * 100, 'theta: ', theta)
                # print('Q_val:', self.Q(oe, theta, f_norm))
                print('difference:', dif / np.dot(self.w_oe.T, self.oe_t))
                print('oe:', oe.T[0])
                ax.plot(self.r_his[0,i], self.r_his[1,i], marker='.', color='blue')

                ax.set_xlabel('X label')
                ax.set_ylabel('Y label')
                # 0.001秒停止
                plt.pause(0.001)

            if (dif / np.dot(self.w_oe.T, self.oe_t)  < 1e-3 ).all():
                print('reach')
                break

     
    def update(self, oe, theta, m, f_vec, t):
        """1stepだけまとめてアップデート

        Args:
            oe (ndarray): 5*1 ([a, e, i, Omega, omega])
            theta (float): true anomaly
            m (float): mass
            f_vec (ndarray): 3*1 ([f_r, f_theta, f_h])
            t (float): time now
        
        Returns:
            ndarray: 5*1 ([a, e, i, Omega, omega])
            float: true anomaly
            float: mass
        """
        def doe_dt_wraped(oe):
            return self.doe_dt(oe, theta, f_vec)
        
        def dtheta_dt_wraped(theta):
            return self.dtheta_dt(oe, theta, f_vec)
        
        def dm_dt_wraped(m):
            return self.dm_dt()
        oe_next = runge_kutta_1step(doe_dt_wraped, oe, self.dt)
        theta_next = runge_kutta_1step(dtheta_dt_wraped, theta, self.dt)
        m_next = runge_kutta_1step(dm_dt_wraped, m, self.dt)
        return oe_next, theta_next, m_next
    
    def calc_input(self, oe, theta, f_norm):
        d1, d2, d3 = self.d123(oe, theta, f_norm)
        min_dQ_dt, max_dQ_dt = self.min_max_dQ_dt(oe, f_norm)
        dQ_dt = self.dQ_dt(oe, theta, f_norm)
        if (self.eta != 0):
            eff = (dQ_dt - max_dQ_dt) / (min_dQ_dt - max_dQ_dt)
            # print(eff, theta)

            if (eff > self.eta):
                input = - f_norm * np.array([d2, d1, d3]) / np.linalg.norm([d1, d2, d3])
                self.dt = self.dt_nominal
            else:
                input = 0 * f_norm * np.array([d2, d1, d3]) / np.linalg.norm([d1, d2, d3]) # あんまよくないけど形を保つため
                self.dt = self.dt_skip
        else:
            input = - f_norm * np.array([d2, d1, d3]) / np.linalg.norm([d1, d2, d3])
        
        if (oe[0] < self.a_threshold):
            self.dt = self.dt_nominal

        return input
    
    # ---------------------------------------------------------------------------
    # 以下はアップデート用の微分値
    # ---------------------------------------------------------------------------
    def min_max_dQ_dt(self, oe, f_norm, search_num = 36):
        theta_grid = np.linspace(0, 2 * np.pi, search_num)
        dQ_dt_list = [0]*search_num

        for i in range(len(theta_grid)):
            theta = theta_grid[i]
            dQ_dt_list[i] = self.dQ_dt(oe, theta, f_norm)[0]

        return min(dQ_dt_list), max(dQ_dt_list)

    def dQ_dt(self, oe, theta, f_norm):
        d1, d2, d3 = self.d123(oe, theta, f_norm)
        alpha = np.arctan2(-d2, -d1)
        beta = np.arctan2(-d3, -d1 / np.cos(alpha) / f_norm)
        res = d1 * np.cos(beta) * np.cos(alpha) + d2 * np.cos(beta) * np.sin(alpha) + d3 * np.sin(beta)
        return res
    
    def d123(self, oe, theta, f_norm):
        dq_doe = self.dq_doe(oe, theta, f_norm)
        doe_df_r = self.ddoe_dtdf_r(oe, theta)
        doe_df_theta = self.ddoe_dtdf_theta(oe, theta)
        doe_df_h = self.ddoe_dtdf_h(oe, theta)

        d1 = np.dot(dq_doe.T, doe_df_theta)[0]
        d2 = np.dot(dq_doe.T, doe_df_r)[0]
        d3 = np.dot(dq_doe.T, doe_df_h)[0]
        return d1, d2, d3
    
    def doe_dt(self, oe, theta, f_vec):
        res = self.ddoe_dtdf_r(oe, theta) * f_vec[0] + self.ddoe_dtdf_theta(oe, theta) * f_vec[1] + self.ddoe_dtdf_h(oe, theta) * f_vec[2]
        return res
    
    def dtheta_dt(self, oe, theta, f_vec):
        a, e, _, _, _ = oe[:,0]
        h = self.h(a, e)
        p = self.p(a, e)
        r = self.r(a, e, theta)

        res = h / r**2 + 1 / e / h *(p * np.cos(theta) * f_vec[0] - (p + r) * np.sin(theta) * f_vec[1])
        # res = [h / r**2]
        return res[0]
    
    def dm_dt(self):
        return - self.thrust / I_sp / g
    
    # ---------------------------------------------------------------------------
    # 以下は計算用の微分値
    # ---------------------------------------------------------------------------
    def ddoe_dtdf_r(self, oe, theta):
        res = np.zeros(oe.shape)

        a, e, _, _, _ = oe[:,0]
        h = self.h(a, e)
        p = self.p(a, e)

        res[0] = 2 * a**2 / h * e * np.sin(theta)
        res[1] = p * np.sin(theta) / h 
        res[2] = 0
        res[3] = 0
        res[4] = - p * np.cos(theta) / h / e

        return res
    
    def ddoe_dtdf_theta(self, oe, theta):
        res = np.zeros(oe.shape)

        a, e, _, _, _ = oe[:,0]
        h = self.h(a, e)
        p = self.p(a, e)
        r = self.r(a, e, theta)

        res[0] = 2 * a**2 / h * p / r
        res[1] = ((p + r) * np.cos(theta) + r * e) / h
        res[2] = 0
        res[3] = 0
        res[4] = (p + r) * np.sin(theta) / h / e

        return res
    
    def ddoe_dtdf_h(self, oe, theta):
        res = np.zeros(oe.shape)

        a, e, i, _, omega = oe[:,0]
        h = self.h(a, e)
        r = self.r(a, e, theta)

        res[0] = 0
        res[1] = 0
        res[2] = r * np.cos(theta + omega) / h
        res[3] = r * np.sin(theta + omega) / h / np.sin(i)
        res[4] = - r * np.sin(theta + omega) * np.cos(i) / h / np.sin(i)

        return res
    
    def dq_doe(self, oe, theta, f_norm):
        w_p = self.w_p
        w_oe = self.w_oe
        V = self.V(oe, theta, f_norm)
        dP = self.dP_doe(oe)
        dV = self.dV_doe(oe, theta, f_norm)
        P = self.P(oe)

        res = w_p * np.dot(w_oe.T, V) * dP + (1 + w_p * P) * np.dot(dV, w_oe)
        # print(V)
        return res
    
    def dV_doe(self, oe, theta, f_norm, doe_ratio=0.00000001):
        """calc derivation of V by oe

        Args:
            oe (ndarray): 5*1 ([a, e, i, Omega, omega])
            theta (double): true anomaly
            doe_ratio (doublw, optional): val for calc derivation. Defaults to 0.001.

        Returns:
            ndarray: 5*5 ([ dV_da.T(1 * 5)
                            dV_de.T
                            dV_di.T
                            dV_dOmega.T
                            dV_domega.T])
        """
        doe = oe * doe_ratio
        doe[doe==0] = doe_ratio
        v = self.V(oe, theta, f_norm)
        res = np.zeros((oe.shape[0], oe.shape[0]))
        # print(v)
        for i in range(len(oe)):
            oe_i = oe.copy()
            oe_i[i] += doe[i]
            v_doe_i = self.V(oe_i, theta, f_norm)
            res[:, i] = ((v_doe_i - v) / doe[i])[:,0]
        # print(res)
        return res.T
    
    def V(self, oe, theta, f_norm):
        res = (self.distance(oe) / self.doe_xx_dt(oe, theta, f_norm))**2 * self.s_oe(oe)
        return res
    
    def dP_doe(self, oe):
        a, e, _, _, _ = oe[:,0]
        res = np.array([[- 100 * (1 - e) / self.r_p_min * self.P(oe), 100 * a / self.r_p_min * self.P(oe), 0, 0, 0]])
        return res.T
    
    def doe_xx_dt(self, oe, theta, f_norm):
        a, e, i, _, omega = oe[:,0]
        h = self.h(a, e)
        p = self.p(a, e)
        r = self.r(a, e, theta)

        cos_theta_xx = ((1 - e**2) / 2 / e**3 + (1 / 4 * ((1 - e**2) / e**3)**2 + 1 / 27)**0.5)**(1/3) \
            - (-(1 - e**2) / 2 / e**3 + (1 / 4 * ((1 - e**2) / e**3)**2 + 1 / 27)**0.5)**(1/3) - 1 / e
        sin_theta_xx = (1 - cos_theta_xx**2)**0.5
        rxx = p / (1 + e * cos_theta_xx)
        b = 0.01

        a_xx = 2 * f_norm * (a**3 * (1 + e) / mu / (1 - e))**0.5
        e_xx = 2 * p * f_norm / h
        i_xx = p * f_norm / h / ((1 - e**2 * np.sin(omega)**2)**0.5 - e * np.abs(np.cos(omega)))
        Omega_xx = p * f_norm / h / np.sin(i) / ((1 - e**2 * np.cos(omega)**2)**0.5 - e * np.abs(np.sin(omega)))

        omega_xxi = f_norm / e / h * (p**2 * cos_theta_xx**2 + (p + rxx)**2 * sin_theta_xx**2)**0.5
        omega_xxo = Omega_xx * np.abs(np.cos(i))
        omega_xx = (omega_xxi + b * omega_xxo) / (1 + b)

        res = np.array([[a_xx, e_xx, i_xx, Omega_xx, omega_xx]])
        return res.T
    
    # ---------------------------------------------------------------------------
    # 以下は補助関数
    # ---------------------------------------------------------------------------
    def Q(self, oe, theta, f_norm):
        res = (1 + self.w_p * self.P(oe)) * np.dot(self.w_oe.T, self.V(oe, theta, f_norm))
        return res[0,0]
    
    def distance(self, oe):
        a, e, i, Omega, omega = oe[:,0]
        a_t, e_t, i_t, Omega_t, omega_t = self.oe_t[:,0]
        # res = np.array([[a - a_t, e - e_t, i - i_t, np.arccos(np.cos(Omega - Omega_t)), np.arccos(np.cos(omega - omega_t))]])
        res = np.array([[a - a_t, e - e_t, i - i_t, (Omega - Omega_t), (omega - omega_t)]])
        return res.T
    
    def s_oe(self, oe):
        a, _, _, _, _ = oe[:,0]
        a_t, _, _, _, _ = self.oe_t[:,0]
        res = np.array([[(1 + ((a - a_t) / 3 / a_t)**4)**0.5, 1, 1, 1, 1]])
        return res.T

    def P(self, oe):
        a, e, _, _, _ = oe[:,0]
        r_p = a * (1 - e)
        res = np.exp(100 * (1 - r_p / self.r_p_min))
        return res
    
    def h(self, a, e):
        p = self.p(a, e)
        h = (p * mu)**0.5
        return h
    
    def p(self, a, e):
        p = a * (1 - e**2)
        return p

    def r(self, a, e, theta):
        p = self.p(a, e)
        r = p / (1 + e * np.cos(theta))
        return r

if __name__ == "__main__":
    main()
