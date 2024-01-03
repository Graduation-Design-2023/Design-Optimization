from libs.lib import Values, TrajectoryCalculator, Planet
import numpy as np
from matplotlib import pyplot as plt

class LambertSolver():
    """
    ランベルト問題を解くための関数群を保持。
    """
    def __init__(self, k_hat_vec = np.array([[0.],[0.],[1.]]),values = Values(), center_planet = 'Sun'):
        self.mu = values.mu(center_planet)
        self.k_hat_vec = k_hat_vec

    def set_variables(self,r1_vec,r2_vec,t1,t2):
        """
        引数--------------------------
        r1_vec : 3*1 ndarray
            出発位置(km)
        r2_vec : 3*1 ndarray
            到着位置(km)
        t1 : double
            出発時刻(s)
        t2 : double
            到着時刻
        """
        self.r1_vec = r1_vec
        self.r2_vec = r2_vec
        self.r1 = np.linalg.norm(r1_vec)
        self.r2 = np.linalg.norm(r2_vec)
        self.t1 = t1
        self.t2 = t2
        self.delta_t = t2 - t1 #飛行時間
        self.c = np.linalg.norm(r2_vec - r1_vec) #2点間距離
        self.s = (self.r1 + self.r2 + self.c) / 2. 
        self.a_m = self.s / 2. #楕円軌道での最小長半径。FとF*が同じ側か判断

    def set_flag_delta_nu_is_under_180(self):
        """
        遷移角が180度以上か判断し、nu_is_under_180をセット
        """
        #delta_nuを求める
        cos_delta_nu = np.dot(self.r1_vec.T, self.r2_vec) / self.r1 / self.r2
        sin_delta_nu = (1 - cos_delta_nu**2.)**0.5
        if (np.dot(np.cross(self.r1_vec.T,self.r2_vec.T), self.k_hat_vec) < 0): #r1からr2への向きが右回りなら符号を逆転
            sin_delta_nu = - sin_delta_nu
        self.delta_nu_rad = np.arctan2(sin_delta_nu, cos_delta_nu) #-180~180
        if (self.delta_nu_rad < 0): #0~360の範囲に
            self.delta_nu_rad = self.delta_nu_rad + np.pi * 2.
        self.delta_nu_is_under_180 = (0 <= self.delta_nu_rad <= np.pi)
    
    def set_flag_is_ellipse(self, delta_nu_is_under_180):
        """
        軌道を判断（放物線は考えない）し、is_ellipseをセット
        """
        if (delta_nu_is_under_180):
            delta_t_p = 1. / 3. * (2. / self.mu)**0.5 * (self.s**1.5 - (self.s - self.c)**1.5) #放物線軌道の遷移時間
            self.is_ellipse = self.delta_t > delta_t_p
        else:
            delta_t_p = 1. / 3. * (2. / self.mu)**0.5 * (self.s**1.5 + (self.s - self.c)**1.5) #放物線軌道の遷移時間
            self.is_ellipse = self.delta_t > delta_t_p

    def set_flag_focus_is_same_side(self, delta_nu_is_under_180):
        """
        楕円軌道で焦点が同じ側にあるか判断し、focus_is_same_sideをセット
        """
        #delta_tを求める
        beta_m_rad = 2 * np.arcsin(((self.s - self.c) / self.s)**0.5)
        if (delta_nu_is_under_180):
            delta_t_m = (self.a_m**3. / self.mu)**0.5 * (np.pi - (beta_m_rad - np.sin(beta_m_rad))) #最小楕円軌道の遷移時間
            self.focus_is_same_side = self.delta_t < delta_t_m #双曲線では不要
        else:
            delta_t_m = (self.a_m**3. / self.mu)**0.5 * (np.pi + (beta_m_rad - np.sin(beta_m_rad))) #最小楕円軌道の遷移時間
            self.focus_is_same_side = self.delta_t > delta_t_m #双曲線では不要
    
    def calc_delta_t(self, a):
        """
        長半径に対して飛行時間を返す関数
        引数--------------------------
        a : double
            長半径(km)
        返り値--------------------------
        delta_t : double
            飛行時間
        """
        #楕円の場合
        if (self.is_ellipse): 
            alpha_rad = 2. * np.arcsin((self.s / 2. / a)**0.5)
            beta_rad = 2. * np.arcsin(((self.s - self.c)/ 2. / a)**0.5)
            if (self.delta_nu_is_under_180):
                if (self.focus_is_same_side):
                    return (a**3. / self.mu)**0.5 * ((alpha_rad - np.sin(alpha_rad)) - (beta_rad - np.sin(beta_rad)))
                else:
                    return (a**3. / self.mu)**0.5 * (2 * np.pi - (alpha_rad - np.sin(alpha_rad)) - (beta_rad - np.sin(beta_rad)))
            else:
                if (self.focus_is_same_side):
                    return (a**3. / self.mu)**0.5 * (2 * np.pi - (alpha_rad - np.sin(alpha_rad)) + (beta_rad - np.sin(beta_rad)))
                else:
                    return (a**3. / self.mu)**0.5 * ((alpha_rad - np.sin(alpha_rad)) + (beta_rad - np.sin(beta_rad)))
        else: 
            gamma_rad = 2. * np.arcsinh((self.s / 2 / a)**0.5)
            delta_rad = 2. * np.arcsinh(((self.s - self.c)/ 2 / a)**0.5)
            if (self.delta_nu_is_under_180):
                return (a**3. / self.mu)**0.5 * ((np.sinh(gamma_rad) - gamma_rad) - (np.sinh(delta_rad) - delta_rad))
            else:
                return (a**3. / self.mu)**0.5 * ((np.sinh(gamma_rad) - gamma_rad) + (np.sinh(delta_rad) - delta_rad))
            
    def calc_d_da_delta_t(self, a):
        """
        飛行時間の長半径微分を返す関数
        引数--------------------------
        a : double
            長半径(km)
        返り値--------------------------
        d_da_delta_t : double
            飛行時間の長半径微分
        """
        T = 2. * np.pi * (a**3. / self.mu)**0.5
        if (self.is_ellipse):
            alpha_rad = 2. * np.arcsin((self.s / 2. / a)**0.5)
            beta_rad = 2. * np.arcsin(((self.s - self.c)/ 2. / a)**0.5)
            if (self.delta_nu_is_under_180):
                if (self.focus_is_same_side):
                    return 3. * self.delta_t / 2. / a - 1 / (self.mu * a**3)**0.5 * (self.s**2 / np.sin(alpha_rad) - (self.c - self.s)**2. / np.sin(beta_rad))
                else:
                    return 3. * T / 2. / a + 3. * self.delta_t / 2. / a + 1 / (self.mu * a**3)**0.5 * (self.s**2 / np.sin(alpha_rad) + (self.c - self.s)**2. / np.sin(beta_rad))
            else:
                if (self.focus_is_same_side):
                    return 3. * T / 2. / a + 3. * self.delta_t / 2. / a + 1 / (self.mu * a**3)**0.5 * (self.s**2 / np.sin(alpha_rad) - (self.c - self.s)**2. / np.sin(beta_rad))
                else:
                    return 3. * self.delta_t / 2. / a - 1 / (self.mu * a**3)**0.5 * (self.s**2 / np.sin(alpha_rad) + (self.c - self.s)**2. / np.sin(beta_rad))
                   
        else:
            gamma_rad = 2 * np.arcsinh((self.s / 2 / a)**0.5)
            delta_rad = 2 * np.arcsinh(((self.s - self.c)/ 2 / a)**0.5)
            if (self.delta_nu_is_under_180):
                return 3. * self.delta_t / 2. / a - 1 / (self.mu * a**3)**0.5 * (self.s**2 / np.sinh(gamma_rad) - (self.c - self.s)**2. / np.sinh(delta_rad))
            else:
                return 3. * self.delta_t / 2. / a - 1 / (self.mu * a**3)**0.5 * (self.s**2 / np.sinh(gamma_rad) + (self.c - self.s)**2. / np.sinh(delta_rad))
                                                                                                                                                
    def calc_a(self):
        """
        ニュートンラフソン法によって飛行時間を満たす長軸半径を求める
        返り値--------------------------
        a : double
            長半径(km)
        """
        a = 1.1 * self.a_m
        n = 0
        while(np.abs(self.calc_delta_t(a) - self.delta_t) > self.delta_t * 10**(-10)):
            rho = 1.
            while(a + rho*(self.delta_t - self.calc_delta_t(a)) / self.calc_d_da_delta_t(a) < self.a_m): #a_m以下にはならないことからステップをつける（あるみほ条件）
                rho = 0.9*rho
            a = a + rho * (self.delta_t - self.calc_delta_t(a)) / self.calc_d_da_delta_t(a)
            n = n + 1
            if(n > 1000):
                print("can't calculate",self.t1,self.t2)
                break
        if (not self.is_ellipse):
            a = -a
        return a

    def solve_lambert(self):
        """
        ランベルト問題を解く関数
        output--------------------------
        a : double
            semi-major axis(sun centered)
        e : double
            eccentricity(sun centered)
        p : double
        nu_1_rad : double
            nu at the orbit insertion(sun centered)
        nu_2_rad : double
            nu when a sat achieves target planet(sun centered)
        r_start : 3*1 ndarray(double)
            position at the orbit insertion(sun centered)
        r_end : 3*1 ndarray(double)
            position when a sat achieves target planet(sun centered)
        v1_vec : 3*1 ndarray(double)
            velocity of a sat at the orbit insetion(sun centered)
        v2_vec : 3*1 ndarray(double)
            velocity of a sat when achieved target planet(sun centered)
        """
        #形状などを判断
        self.set_flag_delta_nu_is_under_180()
        self.set_flag_is_ellipse(self.delta_nu_is_under_180)
        self.set_flag_focus_is_same_side(self.delta_nu_is_under_180)
        #Δtを満たす長半径を求める
        a = self.calc_a()
        
        #以下は後処理
        #pを求める
        if (self.is_ellipse):
            alpha_rad = 2. * np.arcsin((self.s / 2. / a)**0.5)
            beta_rad = 2. * np.arcsin(((self.s - self.c)/ 2. / a)**0.5)
            if(self.focus_is_same_side):
                p = 4 * a * (self.s - self.r1) * (self.s - self.r2) / self.c**2 * np.sin(alpha_rad / 2 + beta_rad / 2)**2
            else:
                p = 4 * a * (self.s - self.r1) * (self.s - self.r2) / self.c**2 * np.sin(alpha_rad / 2 - beta_rad / 2)**2
        else:
            gamma_rad = 2 * np.arcsinh((self.s / 2 / -a)**0.5)
            delta_rad = 2 * np.arcsinh(((self.s - self.c)/ 2 / -a)**0.5)
            if(self.delta_nu_is_under_180):
                p = -4 * a * (self.s - self.r1) * (self.s - self.r2) / self.c**2 * np.sinh(gamma_rad / 2 + delta_rad / 2)**2
            else:
                p = -4 * a * (self.s - self.r1) * (self.s - self.r2) / self.c**2 * np.sinh(gamma_rad / 2 - delta_rad / 2)**2
        #離心率
        e = (1 - p / a)**0.5
        #ν1,2を求める
        cos_nu_1 = (p - self.r1) / 2 / self.r1
        sin_nu_1 = 1 / np.sin(self.delta_nu_rad) * (cos_nu_1 * np.cos(self.delta_nu_rad) - (p - self.r2) / 2 / self.r2) #加法定理（arctanでnu_1を求めたい）
        nu_1_rad = np.arctan2(sin_nu_1,cos_nu_1)
        nu_2_rad = (nu_1_rad + self.delta_nu_rad)
        #v1,v2を求める
        g = self.r1 * self.r2 * np.sin(self.delta_nu_rad) / (self.mu * p)**0.5
        f = 1 - self.r2 / p * (1 - np.cos(self.delta_nu_rad))
        g_dot = 1 - self.r1 / p * (1 - np.cos(self.delta_nu_rad))
        v1_vec = 1 / g * (self.r2_vec - f * self.r1_vec)
        v2_vec = 1 / g * (g_dot * self.r2_vec - self.r1_vec)
        return a, e, p, float(nu_1_rad), float(nu_2_rad),v1_vec,v2_vec
    
class PlanetsTransOrbit():
    def __init__(self, planet_start, planet_end, calculator = TrajectoryCalculator, lambert = LambertSolver, values = Values):
        self.planet_start = planet_start
        self.planet_end = planet_end
        self.calculator_start = TrajectoryCalculator(planet_start.planet_name)
        self.calculator_end = TrajectoryCalculator(planet_end.planet_name)
        self.calculator_sun = TrajectoryCalculator("Sun")
        self.lambert = LambertSolver()
        self.lambert_planet_end = LambertSolver(center_planet=planet_end.planet_name)
        self.values = Values()
    
    def launch_window_period(self, JS):
        """
        calc the period between launch windows
        input--------------------------
        JS : double
            reference time for planet orbital elements
        output--------------------------
        period : double
            the period between launch windows
        """
        JD = JS / 24 / 60 / 60
        T_TDB = (JD - 2451545.0) / 36525.0
        a_start = self.values.a(self.planet_start.planet_name, T_TDB)
        a_end = self.values.a(self.planet_end.planet_name, T_TDB)
        mu = self.values.mu('Sun')
        n_start = (mu / a_start**3)**0.5
        n_end = (mu / a_end**3)**0.5
        period = 2 * np.pi / np.abs(n_start - n_end)
        return period


    def calc_nu_between_planets(self, JS):
        """
        calculate angle between planets at givem JS
        input--------------------------
        JS : double
            time when angle is calculated
        output--------------------------
        nu : double
            angle between planets (0 ~ 360 deg)
        """
        r_vec_start, _ = self.planet_start.position_JS(JS)
        r_vec_end, _ = self.planet_end.position_JS(JS)
        r_vec_start[2] = 0
        r_vec_end[2] = 0
        cos = np.dot(r_vec_start.T, r_vec_end) / np.linalg.norm(r_vec_start) / np.linalg.norm(r_vec_end)
        nu_rad = np.arccos(cos)
        nu_deg = np.rad2deg(nu_rad)
        outer = np.cross(r_vec_start[:,0], r_vec_end[:,0])
        if (outer[2] > 0):
            return nu_deg
        else:
            return -nu_deg + 360
        
    def delta_nu(self, JS):
        JD = JS / 24 / 60 / 60
        T_TDB = (JD - 2451545.0) / 36525.0
        a_start = self.values.a(self.planet_start.planet_name, T_TDB)
        a_end = self.values.a(self.planet_end.planet_name, T_TDB)
        mu = self.values.mu("Sun")
        t_H = np.pi * ((a_start + a_end)**3 / 8 / mu)**0.5
        n_end = (mu / a_end**3)**0.5
        delta_nu_rad = np.pi - n_end * t_H
        return np.rad2deg(delta_nu_rad), t_H
    
    def calc_launch_window(self, year_first, month_first, date_first, threshold, num):
        """
        calculate launch windows by binory search
        input--------------------------
        year_first, month_start, date_start : int
            start time of iteration to calculate launch window
        threshold : double
            threshold of iteration to calculate launch window
        num : double
            number of launch windows you need
        output--------------------------
        sol : num*1 ndarray(double)
            num launch windows
        t_H : double
            time necessary for hohmann transition (s)
        """
        JS_start, _, _ = self.values.convert_times_to_T_TDB(year_first, month_first, date_first, 0, 0, 0)
        period = self.launch_window_period(JS_start)
        JS_last = JS_start + period
        JS_mid = (JS_last + JS_start) / 2

        count = 0

        nu_start = self.calc_nu_between_planets(JS_start)
        delta_nu, t_H = self.delta_nu(JS_start)
        nu_mid = self.calc_nu_between_planets(JS_mid)
        sign_delta_nu = delta_nu / np.abs(delta_nu)

        if (nu_start > delta_nu):
            delta_nu += 360
        if (nu_start > nu_mid):
            nu_mid += 360

        while(np.abs(nu_mid - delta_nu) > threshold):
            if(sign_delta_nu * (nu_mid - delta_nu) > 0):
                JS_start = JS_mid
            else:
                JS_last = JS_mid
            
            JS_mid = (JS_last + JS_start) / 2
            nu_mid = self.calc_nu_between_planets(JS_mid)
            if (nu_start > nu_mid):
                nu_mid += 360
            count += 1
            if(count > 1000):
                print("exceed max num of iteration")
                break
        JS_array = np.arange(JS_mid, JS_mid+num*period, period)
        JD_array = JS_array / 24 / 60 / 60
        sol = np.zeros((num,6))
        for i in range(num):
            sol[i] =  np.array([self.values.convert_JD_to_times(JD_array[i])])
        return sol, t_H
    
    def swingby(self,planet,theta_rad,r_h,v_B_vec,JS):
        """
        input-------------------
        planet : Planet
            planet used for swingby
        theta_rad : double
            angle between B and T_hat
        r_h : double
            distance from planet at closest approach
        v_B_vec : 3*1 ndarray
            v_in(sun centered)
        JS : double
            time starting swingby    
        """
        k_hat_vec = np.array([[0],[0],[1]])
        r_planet_vec,v_planet_vec = planet.position_JS(JS) 
        v_inf_in_vec = v_B_vec - v_planet_vec 
        v_inf = np.linalg.norm(v_inf_in_vec)

        # calc orbital elements(planet centered)
        a = - planet.mu / v_inf**2
        e = 1 + r_h * v_inf**2 / planet.mu
        v_h = (2 * planet.mu / r_h + v_inf**2)**0.5

        # calc b
        phi_B_rad = 2 * np.arcsin(1 / (1 + r_h * v_inf**2 / planet.mu))
        b = r_h * (1 + 2 * planet.mu / r_h / v_inf**2 )

        # calc unit vectors and B vector
        S_I_hat_vec = v_inf_in_vec / v_inf
        lambda_I_rad = np.arctan2(S_I_hat_vec[1],S_I_hat_vec[0])
        if (lambda_I_rad < 0):
            lambda_I_rad = lambda_I_rad + np.pi * 2
        beta_I = np.arcsin(S_I_hat_vec[2])
        T_I_hat_vec = np.array([[S_I_hat_vec[1][0] / (S_I_hat_vec[0][0]**2 + S_I_hat_vec[1][0]**2)**0.5],[-S_I_hat_vec[0][0] / (S_I_hat_vec[0][0]**2 + S_I_hat_vec[1][0]**2)**0.5],[0]])
        R_hat_vec = np.cross(S_I_hat_vec.T,T_I_hat_vec.T).T
        B_vec = np.array([[b / (S_I_hat_vec[0][0]**2 + S_I_hat_vec[1][0]**2)**0.5 * (S_I_hat_vec[1][0]*np.cos(theta_rad) + S_I_hat_vec[0][0]*S_I_hat_vec[2][0]*np.sin(theta_rad))],
                          [b / (S_I_hat_vec[0][0]**2 + S_I_hat_vec[1][0]**2)**0.5 * (-S_I_hat_vec[0][0]*np.cos(theta_rad) + S_I_hat_vec[1][0]*S_I_hat_vec[2][0]*np.sin(theta_rad))],
                          [- b * (S_I_hat_vec[0][0]**2 + S_I_hat_vec[1][0]**2)**0.5 * np.sin(theta_rad)]])
        v_inf_out_vec = v_inf * (np.cos(phi_B_rad) * S_I_hat_vec - np.sin(phi_B_rad) * B_vec / b)

        # calc out direction
        S_O_hat_vec = v_inf_out_vec / v_inf
        W_hat_vec = np.cross(S_I_hat_vec.T,S_O_hat_vec.T).T / np.linalg.norm(np.cross(S_I_hat_vec.T,S_O_hat_vec.T).T)
        N_hat_vec = np.cross(k_hat_vec.T,W_hat_vec.T).T / np.linalg.norm(np.cross(k_hat_vec.T,W_hat_vec.T))

        # calc remaining orbital elements
        i = np.arccos(np.dot(W_hat_vec.T,k_hat_vec))
        Omega = np.arctan2(- W_hat_vec[1][0]/(W_hat_vec[0][0]**2 + W_hat_vec[1][0]**2)**0.5, W_hat_vec[0][0]/(W_hat_vec[0][0]**2 + W_hat_vec[1][0]**2)**0.5)
        if (Omega < 0):
            Omega = Omega + 2 * np.pi
        r_h_vec = r_h  * (np.sin(phi_B_rad/2) * S_I_hat_vec + np.cos(phi_B_rad) * B_vec / b)
        v_h_vec = v_h  * (np.cos(phi_B_rad/2) * S_I_hat_vec - np.sin(phi_B_rad) * B_vec / b)
        omega = np.arctan2(np.dot(N_hat_vec.T,r_h_vec)/r_h, np.dot(np.cross(W_hat_vec.T, N_hat_vec.T), r_h_vec)/r_h)
        oes = (a,e,i,omega,Omega)
        return v_inf_out_vec, v_inf_out_vec+v_planet_vec, r_h_vec, v_h_vec, oes
    
    def trajectory_by_lambert(self, time_start, duration):
        """
        calc the trajectory between planets by solving lambert
        input--------------------------
        time_start : 6 tuple(double)
            time at orbit insertion(y,m,d,h,m,s)
        duration : double
            time to trans(s)
        output--------------------------
        nu_1_rad : double
            nu at the orbit insertion(sun centered)
        nu_2_rad : double
            nu when a sat achieves target planet(sun centered)
        r_start : 3*1 ndarray(double)
            position at the orbit insertion(sun centered)
        r_end : 3*1 ndarray(double)
            position when a sat reaches target planet(sun centered)
        v_planet_start : 3*1 ndarray(double)
            velocity of a planet at the orbit insetion(sun centered)
        v_planet_end : 3*1 ndarray(double)
            velocity of a planet when reached target planet(sun centered)
        v_sat_start : 3*1 ndarray(double)
            velocity of a sat at the orbit insetion(sun centered)
        v_sat_end : 3*1 ndarray(double)
            velocity of a sat when reached target planet(sun centered)
        """
        JS_start, _, _ = self.values.convert_times_to_T_TDB(*time_start)
        JS_end = JS_start + duration
        r_start, v_planet_start = self.planet_start.position_JS(JS_start)
        r_end, v_planet_end = self.planet_end.position_JS(JS_end)
        self.lambert.set_variables(r_start, r_end, JS_start, JS_end)
        _, _, _, nu_1_rad, nu_2_rad,v_sat_start,v_sat_end = self.lambert.solve_lambert()
        delta_v1 = np.linalg.norm(v_planet_start - v_sat_start)
        delta_v2 = np.linalg.norm(v_planet_end - v_sat_end)
        return nu_1_rad, nu_2_rad,r_start,r_end,v_planet_start,v_planet_end,v_sat_start,v_sat_end
    
    def trajectory_with_1TCM(self, time_start, time_end, time_tcm, v_inf_end):
        """
        calc the trajectory which reaches given v_inf_end with 1 TCM
        input--------------------------
        time_start : 6 tuple(double)
            time at the orbit insertion(y,m,d,h,m,s)
        time_end : 6 tuple(double)
            time when a sat reaches target planet(y,m,d,h,m,s)
        time_tcm : 6 tuple(double)
            time at the tcm(y,m,d,h,m,s)
        v_inf_end : 3*1 ndarray(double)
            velocity when a sat reached the target planet(taget planet centered)
        output-------------------------
        r_start
        v_sat_start
        v_planet_start
        r_tcm
        v_sat_before_tcm
        v_sat_aftert_tcm
        r_end
        v_sat_end
        v_planet_end
        """
        JS_start, _, _ = self.values.convert_times_to_T_TDB(self, *time_start)
        JS_end, _, _ = self.values.convert_times_to_T_TDB(self, *time_end)
        JS_tcm, _, _ = self.values.convert_times_to_T_TDB(self, *time_tcm)
        r_start, v_planet_start = self.planet_start.position_JS(JS_start)
        r_end, v_planet_end = self.planet_end.position_JS(JS_end)
        # after tcm
        v_sat_end = v_planet_end + v_inf_end
        r_tcm, v_sat_after_tcm = self.calculator_sun.calc_r_v_form_r_v_0(r_end, v_sat_end, JS_end, JS_tcm)
        # before tcm
        self.lambert.set_variables(r_start, r_tcm, JS_start, JS_tcm)
        _, _, _, nu_1_rad, nu_2_rad,v_sat_start,v_sat_before_tcm = self.lambert.solve_lambert()
        v_inf_start = v_sat_start - v_planet_start
        return r_start, v_sat_start, v_planet_start,r_tcm, v_sat_before_tcm, v_sat_after_tcm, r_end, v_sat_end, v_planet_end
    
    def trajectory_insertion01(self,theta,r_h,v_in_vec,JS,r_a_planetary,oe_observation):
        """
        use coapsidal capture orbit for insertion into planetary orbit(1)
        input-------------------
        theta : double
            angle between B and T_hat
        r_h : double
            distance from planet at closest approach
        v_in_vec : 3*1 ndarray
            v_in(sun centered)
        JS : double
            time entering target planet gravity field   
        r_a_planetary : double
            apogee distance for coapsidal capture orbit
        output-------------------
        r_01_vec
        r_12_vec
        v0_end_vec
        v1_start_vec
        v1_end_vec
        0(nu1_start)
        nu1_end 
        JS0_end 
        JS1_end
        """
        # values of trans trajectory"
        theta_rad = np.deg2rad(theta)
        _,_,r_01_vec, v0_end_vec, oes = self.swingby(self.planet_end,theta_rad, r_h, v_in_vec, JS)
        _,_,i,omega,Omega = oes
        r_01 = np.linalg.norm(r_01_vec)
        v0_end = np.linalg.norm(v0_end_vec)

        # values of planetary orbit
        v1_start = (self.planet_end.mu * (2/r_h - 2/(r_h + r_a_planetary)))**0.5
        v1_start_vec = v1_start * v0_end_vec / v0_end

        JS0_end = JS #fix me

        r_12_vec , v1_end_vec, nu1_end = self.calc_point_where_change_plane(oe_observation, r_01_vec, v1_start_vec, JS0_end)
        a1, e1, i1, omega1, Omega1, t_p1, P_hat1_vec, Q_hat1_vec, W_hat1_vec = self.calculator_end.calc_orbital_elems_from_r_v(r_01_vec, v1_start_vec, JS0_end)
        JS1_end = self.calculator_end.calc_time_from_r_vec(r_12_vec, P_hat1_vec, Q_hat1_vec, a1, e1, t_p1)
        # print(JS1_end - t_p1)
        # v1_end_vec, r_debug = self.calculator_end.calc_r_v_form_r_v_0(r_01_vec, v1_start_vec, JS0_end, JS1_end)

        return r_01_vec, r_12_vec, v0_end_vec, v1_start_vec, v1_end_vec, 0, nu1_end, JS0_end, JS1_end
    
    def calc_point_where_change_plane(self, oe_observation, r_01_vec, v1_start_vec, JS0_end):
        """
        calc point where change plane from planetary plane to observation plane
        input----------------------
        oe_observation : 6 tuple
            orbital elements of observation orbit
        r_p1_vec : 3*1 ndarray
            position at perigee  of planetary orbit
        v_p1 vec
            velocity at perigee  of planetary orbit
        JS_p1 : double
            time at perigee  of planetary orbit
        output---------------------
        r_n_vec
        v_n_vec
        nu_n
        """
        a1, e1, i1, omega1, Omega1, t_p1, P_hat1_vec, Q_hat1_vec, W_hat1_vec = self.calculator_end.calc_orbital_elems_from_r_v(r_01_vec, v1_start_vec, JS0_end)
        p1 = a1 * (1 - e1**2)
        _, _, i3, _, Omega3, _ = oe_observation
        P_hat2_vec, Q_hat2_vec, W_hat2_vec = self.calculator_end.calc_PQW_from_orbital_elems(*oe_observation)
        i1_rad = np.deg2rad(i1)
        i3_rad = np.deg2rad(i3)
        Omega1_rad = np.deg2rad(Omega1)
        Omega3_rad = np.deg2rad(Omega3)

        val_x1 = - np.cos(i3_rad) * np.cos(Omega1_rad) * np.sin(i1_rad) + np.cos(Omega3_rad) * np.sin(i3_rad) * np.cos(i1_rad)
        val_x2 = - np.sin(Omega3_rad) * np.sin(i3_rad) * np.cos(Omega1_rad) * np.sin(i1_rad) + np.sin(Omega1_rad) * np.sin(i1_rad) * np.cos(Omega3_rad) * np.sin(i3_rad)
        val_y1 = np.sin(Omega3_rad) * np.sin(i3_rad) * np.cos(i1_rad) - np.cos(i3_rad) * np.sin(Omega1_rad) * np.sin(i1_rad)
        val_y2 = - np.sin(Omega3_rad) * np.sin(i3_rad) * np.cos(Omega1_rad) * np.sin(i1_rad) + np.sin(Omega1_rad) * np.sin(i1_rad) * np.cos(Omega3_rad) * np.sin(i3_rad)
        
        r_n_hat_vec = np.array([[- val_x1 / val_x2], [- val_y1 / val_y2], [1]])
        
        r_n_hat = np.linalg.norm(r_n_hat_vec)
        e_n_vec = r_n_hat_vec / r_n_hat
        r_01 = np.linalg.norm(r_01_vec)
        cos_nu_n = np.dot(e_n_vec.T, r_01_vec) / r_01

        r_n_p = p1 / (1 + e1 * cos_nu_n)
        r_n_m = p1 / (1 - e1 * cos_nu_n)
        if (r_n_p > r_n_m):
            r_n = r_n_p
            r_n_vec = r_n * e_n_vec
        else:
            r_n = r_n_m
            r_n_vec = - r_n * e_n_vec
        P_hat1_vec, Q_hat1_vec, W_hat1_vec = self.calculator_end.calc_PQW_from_orbital_elems(a1,e1,i1,omega1,Omega1,t_p1)
        nu_n = self.calculator_end.calc_nu_from_r_vec(r_n_vec, P_hat1_vec, Q_hat1_vec)
        print("nu_n:", nu_n)
        r_n_ref_vec, v_n_vec = self.calculator_end.calc_rv_from_nu(nu_n, a1, e1, P_hat1_vec, Q_hat1_vec)
        print((self.planet_end.mu * (2 / r_n[0,0] - 1 / a1))**0.5, np.linalg.norm(v_n_vec))
        print(np.dot(W_hat1_vec.T, v_n_vec))
        return r_n_vec, v_n_vec, nu_n
    
    def trajectory_insertion12(self, oe_observation, r_12_vec, JS1_end, target_is_perigee):
        """
        trajectory insertion12
        input--------------------
        oe_observation : 6 tuple
            orbital elements of observation orbit
        r_12_vec : 3*1 ndarray
            position at maneuver time of insertion12
        JS1_end : double
            maneuver time of insertion12
        target_is_perigee : bool
        output------------------
        r_23_vec
        v2_start_vec
        v2_end_ve
        nu2_start
        0(nu2_end)
        JS2_end
        """
        # calc  values of observation orbit
        a3, e3, i3, omega3, Omega3, _ = oe_observation
        P_hat3_vec, Q_hat3_vec, W_hat3_vec = self.calculator_end.calc_PQW_from_orbital_elems(*oe_observation)
        r_p3 = a3 * (1 - e3)
        r_a3 = a3 * (1 + e3)
        mu = self.planet_end.mu

        # calc  values concerning r_N
        r_12 = np.linalg.norm(r_12_vec)
        x = np.dot(r_12_vec.T, P_hat3_vec)
        y = np.dot(r_12_vec.T, Q_hat3_vec)
        z = np.dot(r_12_vec.T, W_hat3_vec)
        nu2_start_rad = np.arctan2(y, x)
        nu2_start = np.rad2deg(nu2_start_rad)
        cos_nu = np.cos(nu2_start_rad)
        self.target_is_perigee = target_is_perigee
        # value when perigee of orbit3 is used
        r_23_p = r_p3
        #fix me
        if (x * (1 - cos_nu) / (r_p3 * cos_nu / mu * (r_p3 - x)) < 0):
            self.target_is_perigee = False
        else:
            v_2_end_p = (x * (1 - cos_nu) / (r_p3 * cos_nu / mu * (r_p3 - x)))**0.5
            r_23_vec_p =  r_23_p * P_hat3_vec
            v2_end_vec_p = v_2_end_p * Q_hat3_vec

        # value when apogee of orbit3 is used
        r_23_a = r_a3
        if (x * (1 + cos_nu) / (r_a3 * cos_nu / mu * (r_a3 + x)) < 0):
            self.target_is_perigee = True
        else:
            v_2_end_a = (x * (1 + cos_nu) / (r_a3 * cos_nu / mu * (r_a3 + x)))**0.5
            r_23_vec_a = - r_23_a * P_hat3_vec
            v2_end_vec_a = - v_2_end_a * Q_hat3_vec

        if(self.target_is_perigee):
            r_23_vec = r_23_vec_p
            v2_end_vec = v2_end_vec_p
            if (r_23_p * v_2_end_p**2 / mu > 1):
                nu2_end = 0
            else:
                nu2_end = 180
                nu2_start += 180
        else:
            r_23_vec = r_23_vec_a
            v2_end_vec = v2_end_vec_a
            if (r_23_a * v_2_end_a**2 / mu > 1):
                nu2_end = 0
                nu2_start += 180
            else:
                nu2_end = 180
        
        # determine orbit2    
        a2, e2, i2, omega2, Omega2, _, P_hat2_vec, Q_hat2_vec, _ = self.calculator_end.calc_orbital_elems_from_r_v(r_23_vec, v2_end_vec, 0) # tp is invalid because JS is wrong value
        delta_t = self.calculator_end.calc_time_from_r_vec(r_12_vec, P_hat2_vec, Q_hat2_vec, a2, e2, 0)
        period2 = self.calculator_end.calc_period(a2)
        t_p2 = JS1_end - delta_t
        debug, v2_start_vec = self.calculator_end.calc_r_v_from_orbital_elems(a2,e2,i2,omega2,Omega2,t_p2,JS1_end)
        JS2_end = t_p2 + period2
    
        return r_23_vec, v2_start_vec, v2_end_vec, nu2_start, nu2_end, JS2_end
        

    def trajectory_insertion12_old(self, oe_observation, r_12_vec, v1_end_vec, JS_p1, duration):
        """
        trajectory insertion2
        input--------------------
        oe_observation : 6 tuple
            orbital elements of observation orbit
        r_p1_vec : 3*1 ndarray
            position at perigee  of planetary orbit
        v_p1 vec
            velocity at perigee  of planetary orbit
        JS_p1 : double
            time at perigee  of planetary orbit
        duration : double
            time for insertion2(s)
        output--------------------
        delta_v_vec
        r_23_vec
        v2_start_vec
        v2_end_vec
        nu2_start
        nu2_end
        """

        a1, e1, i1, omega1, Omega1, t_p1, P_hat1_vec, Q_hat1_vec, _ = self.calculator_end.calc_orbital_elems_from_r_v(r_12_vec, v1_end_vec, JS_p1)
        a3, e3, i3, omega3, Omega3, _ = oe_observation

        # calc start point
        JS_start = self.calculator_end.calc_time_from_r_vec(r_12_vec, P_hat1_vec, Q_hat1_vec, a1, e1, t_p1)

        # calc end point
        P_hat3_vec, _, _ = self.calculator_end.calc_PQW_from_orbital_elems(*oe_observation)
        r_p3_vec = a3 * (1 - e3) * P_hat3_vec
        r_a3_vec = - a3 * (1 + e3) * P_hat3_vec
        if (np.dot(r_12_vec.T, P_hat3_vec) > 0):
            r_23_vec = r_p3_vec
        else:
            r_23_vec = r_a3_vec
        JS_end = JS_start + duration

        # solve lambert
        self.lambert_planet_end.set_variables(r_12_vec, r_23_vec, JS_start, JS_end)
        _, _, _, nu2_start_rad, nu2_end_rad, v2_start_vec, v2_end_vec = self.lambert_planet_end.solve_lambert()
        nu2_start = np.rad2deg(nu2_start_rad)
        nu2_end = np.rad2deg(nu2_end_rad)

        delta_v_vec = v2_start_vec - v1_end_vec
        return delta_v_vec, r_23_vec, v2_start_vec, v2_end_vec, nu2_start, nu2_end

    def trajectory_insertion23(self, oe_observation, r_23_vec, v2_end_vec, JS2_end,target_is_perigee):
        """
        trajectory insertion3
        input--------------------
        oe_observation : 6 tuple
            orbital elements of observation orbit
        r2_end_vec : 3*1 ndarray
            position at the end of insertion2
        v2_end_vec vec
            velocity at the end of insertion2
        JS : double
            time at the end of insertion2
        target_is_perigee : bool
        output-------------------
        v3_start_vec
        nu3_start
        """
        # calc t_p
        a3, e3, i3, omega3, Omega3, _ = oe_observation
        n_rad = (self.planet_end.mu / a3**3)**0.5
        if(self.target_is_perigee):
            t_p3 = JS2_end
        else:
            t_p3 = JS2_end + np.pi / n_rad

        r3_vec, v3_start_vec = self.calculator_end.calc_r_v_from_orbital_elems(a3, e3, i3, omega3, Omega3, t_p3, JS2_end) #fix me
        _, _, _, _, _, _, P_hat3_vec, Q_hat3_vec, _ = self.calculator_end.calc_orbital_elems_from_r_v(r3_vec, v3_start_vec, JS2_end)
        nu3_start = self.calculator_end.calc_nu_from_r_vec(r3_vec, P_hat3_vec, Q_hat3_vec)
        return v3_start_vec, nu3_start

    def trajectory_insertion(self, theta, r_h, v_in_vec, JS0_start, r_a_planetary, oe_observation, target_is_perigee, plot_is_enabled = True):

        r_01_vec, r_12_vec, v0_end_vec, v1_start_vec, v1_end_vec, nu1_start, nu1_end, JS0_end, JS1_end = self.trajectory_insertion01(theta, r_h, v_in_vec, JS0_start, r_a_planetary, oe_observation)
        r_23_vec, v2_start_vec, v2_end_vec, nu2_start, nu2_end, JS2_end = self.trajectory_insertion12(oe_observation, r_12_vec, JS1_end, target_is_perigee)
        v3_start_vec, nu3_start = self.trajectory_insertion23(oe_observation, r_23_vec, v2_end_vec, JS2_end, target_is_perigee)

        delta_v01 = np.linalg.norm(v0_end_vec - v1_start_vec)
        delta_v12 = np.linalg.norm(v1_end_vec - v2_start_vec)
        delta_v23 = np.linalg.norm(v2_end_vec - v3_start_vec)
        delta_v_tot = delta_v01 + delta_v12 + delta_v23

        if (plot_is_enabled):
            fig = plt.figure()
            ax = fig.add_subplot(111,projection = '3d')
            ax.plot(r_12_vec[0],r_12_vec[1],r_12_vec[2], '.')
            self.calculator_end.plot_trajectory(r_01_vec, v0_end_vec, JS0_start, -70, 0, ax)
            self.calculator_end.plot_trajectory(r_01_vec, v1_start_vec, JS0_end, nu1_start, nu1_end, ax, 'k')
            self.calculator_end.plot_trajectory(r_12_vec, v1_end_vec, JS0_end, nu1_start, nu1_end, ax, 'k')
            self.calculator_end.plot_trajectory(r_12_vec, v2_start_vec, JS1_end, nu2_start, nu2_end, ax, 'r')
            self.calculator_end.plot_trajectory(r_23_vec, v3_start_vec, JS2_end, nu3_start, nu3_start+300, ax, 'b')
            plt.show()

        return delta_v_tot,delta_v01 ,delta_v12 ,delta_v23