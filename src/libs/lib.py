import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

class Values():
    """
    定数値など(ソースは大体wikipedia)
    """
    def __init__(self):
        """
        天文単位のみもっておく
        """
        #天文単位(km)
        self.AU = 1.49597870 * 10**8
    
    def convert_times_to_T_TDB(self,year,month,date,hour,minute,second):
        """
        太陽系力学時をユリウス秒、日、世紀に変換
        引数--------------------------
        year : int
            太陽系力学時
        month : int
            太陽系力学時
        date : int
            太陽系力学時
        返り値--------------------------
        JS : double
            ユリウス日を秒単位に換算したもの
        JD : double
            ユリウス日
        T_TDB : double
            ユリウス世紀
        """
        JD  = 367 * year - int(7*(year + int((month + 9)/12))/4) + int(275 * month / 9) + date + 1721013.5 + ((second/60 + minute) / 60 + hour) / 24
        JS = JD * 24 * 60 * 60
        T_TDB = (JD - 2451545.0) / 36525.0
        return JS, JD, T_TDB

    def convert_JD_to_times(self,JD):
        MJD = JD - 2400000.5
        a = int(MJD) + 2400001
        q = MJD - int(MJD)
        b = int((a - 1867216.25)/36524.25)
        c = a +b - int(b/4) + 1525
        d = int((c-121.1)/365.25)
        e = int(365.25*d)
        f = int((c-e)/30.6001)
        D = c - e - int(30.6001*f)
        M = f - 1 - 12 * int(f/14)
        Y = d - 4715 - int((7+M)/10)
        h = int(24 * q)
        m = 60 * (24*q - h)
        s = 60 * (m - int(m))
        return Y,M,D,h,m,s
    
    #長半径(km)
    def a(self, planet_name,T_TDB):
        a_dic = {
            "Mercury":  0.387098310 * self.AU, 
            "Venus"  :  0.723329820 * self.AU, 
            "Earth"  :  1.000001018 * self.AU, 
            "Mars"   :  1.523679342 * self.AU,
            "Jupiter":  5.202603191 * self.AU + 0.0000001913 * T_TDB * self.AU,
            "Saturn" :  9.554909596 * self.AU - 0.0000021389 * T_TDB * self.AU,
            "Uranus" : 19.218446062 * self.AU - 0.0000000372 * T_TDB * self.AU + 0.00000000098 * T_TDB**2. * self.AU,
            "Neptune": 30.110386869 * self.AU + 0.0000001913 * T_TDB * self.AU + 0.00000000069 * T_TDB**2. * self.AU,}
        return a_dic[planet_name]
    
    #離心率
    def e(self, planet_name, T_TDB):
        e_dic = {
            "Mercury": 0.20563175 + 0.000020406 * T_TDB - 0.0000000284 * T_TDB**2. - 0.00000000017 * T_TDB**3.,
            "Venus"  : 0.00677188 - 0.000047766 * T_TDB + 0.0000000975 * T_TDB**2. + 0.00000000044 * T_TDB**3.,
            "Earth"  : 0.01670862 - 0.000042037 * T_TDB - 0.0000001236 * T_TDB**2. + 0.00000000004 * T_TDB**3.,
            "Mars"   : 0.09340062 + 0.000090483 * T_TDB - 0.0000000806 * T_TDB**2. - 0.00000000035 * T_TDB**3.,
            "Jupiter": 0.04849485 + 0.000163244 * T_TDB - 0.0000004719 * T_TDB**2. - 0.00000000197 * T_TDB**3.,
            "Saturn" : 0.05550962 - 0.000346818 * T_TDB - 0.0000006456 * T_TDB**2. + 0.00000000338 * T_TDB**3.,
            "Uranus" : 0.04629590 - 0.000027337 * T_TDB + 0.0000000790 * T_TDB**2. + 0.00000000025 * T_TDB**3.,
            "Neptune": 0.00898809 + 0.000006408 * T_TDB - 0.0000000008 * T_TDB**2.}
        return e_dic[planet_name]
    
    #軌道傾斜角(deg)
    def i(self, planet_name, T_TDB):
        i_dic = {
            "Mercury": 7.004986 - 0.0059516 * T_TDB + 0.00000081 * T_TDB**2. + 0.000000041 * T_TDB**3.,
            "Venus"  : 3.394662 - 0.0008568 * T_TDB - 0.00003244 * T_TDB**2. + 0.000000010 * T_TDB**3.,
            "Earth"  : 0.000000 + 0.0130546 * T_TDB - 0.00000931 * T_TDB**2. - 0.000000034 * T_TDB**3.,
            "Mars"   : 1.849726 - 0.0081479 * T_TDB - 0.00002255 * T_TDB**2. - 0.000000027 * T_TDB**3.,
            "Jupiter": 1.303270 - 0.0019872 * T_TDB + 0.00003318 * T_TDB**2. + 0.000000092 * T_TDB**3.,
            "Saturn" : 2.488878 + 0.0025515 * T_TDB - 0.00004903 * T_TDB**2. + 0.000000018 * T_TDB**3., 
            "Uranus" : 0.773196 - 0.0016869 * T_TDB + 0.00000349 * T_TDB**2. + 0.000000016 * T_TDB**3., 
            "Neptune": 1.769952 + 0.0002257 * T_TDB + 0.00000023 * T_TDB**2. - 0.000000000 * T_TDB**3.,}
        return i_dic[planet_name]
    
    #昇交点経度(deg)
    def Omega(self, planet_name, T_TDB):
        Omega_dic = {
            "Mercury":  48.330893 - 0.1254229 * T_TDB - 0.00008833 * T_TDB**2. - 0.000000196 * T_TDB**3.,
            "Venus"  :  76.679920 - 0.2780080 * T_TDB - 0.00014256 * T_TDB**2. - 0.000000198 * T_TDB**3.,
            "Earth"  :   0.0,
            "Mars"   :  49.558093 - 0.2949846 * T_TDB - 0.00063995 * T_TDB**2. - 0.000002143 * T_TDB**3.,
            "Jupiter": 100.464441 + 0.1766828 * T_TDB + 0.00090387 * T_TDB**2. - 0.000007032 * T_TDB**3.,
            "Saturn" : 113.665524 - 0.2566649 * T_TDB - 0.00018345 * T_TDB**2. + 0.000000357 * T_TDB**3., 
            "Uranus" :  74.005947 + 0.0893206 * T_TDB - 0.00009470 * T_TDB**2. + 0.000000413 * T_TDB**3., 
            "Neptune": 131.784057 - 0.0061651 * T_TDB - 0.00000219 * T_TDB**2. - 0.000000078 * T_TDB**3.}
        return Omega_dic[planet_name]
    
    #近地点引数(deg)
    def omega(self, planet_name,T_TDB):
        omega_dic = {
            "Mercury":  77.456119 + 0.1588643 * T_TDB - 0.00001343 * T_TDB**2. + 0.000000039 * T_TDB**3.- self.Omega("Mercury", T_TDB),
            "Venus"  : 131.563707 + 0.0048646 * T_TDB - 0.00138232 * T_TDB**2. - 0.000005332 * T_TDB**3.- self.Omega("Venus", T_TDB),
            "Earth"  : 102.937348 + 0.3225557 * T_TDB + 0.00015026 * T_TDB**2. + 0.000000478 * T_TDB**3.- self.Omega("Earth", T_TDB),
            "Mars"   : 336.060234 + 0.4438898 * T_TDB - 0.00017321 * T_TDB**2. + 0.000000300 * T_TDB**3.- self.Omega("Mars", T_TDB),
            "Jupiter":  14.331309 + 0.2155525 * T_TDB + 0.00072252 * T_TDB**2. - 0.000004590 * T_TDB**3.- self.Omega("Jupiter", T_TDB),
            "Saturn" :  93.056787 + 0.5665496 * T_TDB + 0.00052809 * T_TDB**2. + 0.000004882 * T_TDB**3.- self.Omega("Saturn", T_TDB),
            "Uranus" : 173.005159 + 0.0893206 * T_TDB - 0.00009470 * T_TDB**2. + 0.000000413 * T_TDB**3.- self.Omega("Uranus", T_TDB),
            "Neptune": 131.784057 + 0.0291587 * T_TDB + 0.00007051 * T_TDB**2. - 0.000000000 * T_TDB**3.- self.Omega("Neptune", T_TDB)}
        return omega_dic[planet_name]
    
    #平均経度(deg)
    def lambdaM(self, planet_name, T_TDB):
        lambdaM_dic = {
            "Mercury": 252.250906 + 149472.6746358 * T_TDB - 0.00000535 * T_TDB**2. + 0.000000002 * T_TDB**3.,
            "Venus"  : 181.979801 +  58517.8156760 * T_TDB + 0.00000165 * T_TDB**2. - 0.000000002 * T_TDB**3.,
            "Earth"  : 100.466449 +  35999.3728519 * T_TDB - 0.00000568 * T_TDB**2. + 0.000000000 * T_TDB**3.,
            "Mars"   : 355.433275 +  19140.2993313 * T_TDB + 0.00000261 * T_TDB**2. - 0.000000003 * T_TDB**3.,
            "Jupiter":  34.351484 +   3034.9056746 * T_TDB - 0.00008501 * T_TDB**2. + 0.000000004 * T_TDB**3.,
            "Saturn" :  50.077471 +   1222.1137943 * T_TDB + 0.00021004 * T_TDB**2. - 0.000000019 * T_TDB**3.,
            "Uranus" : 314.055005 +    428.4669983 * T_TDB - 0.00000486 * T_TDB**2. + 0.000000006 * T_TDB**3., 
            "Neptune": 304.348665 +    218.4862002 * T_TDB + 0.00000059 * T_TDB**2. - 0.000000002 * T_TDB**3.}
        return lambdaM_dic[planet_name]
    
    #重力定数(km^3/s^2) 
    def mu(self,planet_name):
        mu_dic = {
        "Sun"    : 1.32712440 * 10**11,
        "Mercury": 2.203208 * 10**4,
        "Venus"  : 3.248587 * 10**5,
        "Earth"  : 3.986004 * 10**5,
        "Mars"   : 4.282829 * 10**4,
        "Jupiter": 1.267126 * 10**8,
        "Saturn" : 3.793952 * 10**7, 
        "Uranus" : 5.780159 * 10**6, 
        "Neptune": 6.871308 * 10**6}
        return mu_dic[planet_name]
    
    #半径(km)
    def radius(self, planet_name):
        radius_dic = {
        "Sun"    : 6.96 * 10**5,
        "Mercury": 2.4394 * 10**3,
        "Venus"  : 6.0518 * 10**3,
        "Earth"  : 6.378137 * 10**3,
        "Mars"   : 3.39619 * 10**3,
        "Jupiter": 7.1492 * 10**4,
        "Saturn" : 6.0268 * 10**4, 
        "Uranus" : 2.5559 * 10**4, 
        "Neptune": 2.4764 * 10**4}
        return radius_dic[planet_name]

    #赤道の自転周期(s)
    def rotation_period(self, planet_name):
        rotation_period_dic = {
        "Sun"    : 25.379995 * 60**2 * 24,
        "Mercury": 58.6462 * 60**2 * 24,
        "Venus"  : -243.0187 * 60**2 * 24,
        "Earth"  : 0.99726968 * 60**2 * 24,
        "Mars"   : 1.02595675 * 60**2 * 24,
        "Jupiter": 0.41007 * 60**2 * 24,
        "Saturn" : 0.426 * 60**2 * 24,
        "Uranus" : -0.71833 * 60**2 * 24,
        "Neptune": 0.67125 * 60**2 * 24}
        return rotation_period_dic[planet_name]
        
class TrajectoryCalculator(): 
    """
    軌道を求めるための計算機。中心天体ごとにインスタンスを作ること。
    Attributes
    ----------
    values : class
       定数値の保持
    threshold : double
       ニュートンラフソン法の閾値
    mu : double
        重力定数(m^3/s^2)
    """

    def __init__(self, center_planet_name, threshold = 0.01, values = Values()):
        self.values = values
        self.threshold = threshold
        self.mu = self.values.mu(center_planet_name)

    def solve_Kepler(self,a,e,t_p,t,E0):
        """
        ニュートンラフソン法でケプラー方程式を解く
        引数--------------------------
        a : double
            長半径(km)
        e : double
            離心率
        t_p : double
            近地点通過時刻(s)
        mu : double
            重力定数（惑星か太陽かに注意）
        t : double
            r,vを求めたい時刻(s)
        E0 : double
            ニュートンラフソン法の初期値(deg)
        返り値--------------------------
         E : double
            時刻tのEの値(deg)
        """
        E_pre_rad = np.radians(E0)
        E_rad = E_pre_rad - (E_pre_rad - e * np.sin(E_pre_rad) - (self.mu / a**3)**0.5 *(t - t_p)) / (1 - e * np.cos(E_pre_rad))
        while(np.abs(E_pre_rad - E_rad) >= self.threshold):
            E_pre_rad = E_rad
            E_rad = E_pre_rad - (E_pre_rad - e * np.sin(E_pre_rad) - (self.mu / a**3)**0.5 *(t - t_p)) / (1 - e * np.cos(E_pre_rad))
        return np.degrees(E_rad)
    
    def solve_hyperbola_eq(self,a,e,t_p,t,H0):
        """
        ニュートンラフソン法で双曲線の方程式を解く
        引数--------------------------
        a : double
            長半径(km)
        e : double
            離心率
        t_p : double
            近地点通過時刻(s)
        t : double
            時刻(s)
        H0 : double
            ニュートンラフソン法の初期値(deg)
        返り値--------------------------
         H : double
            時刻tのHの値(deg)
        """
        H_pre_rad = np.radians(H0)
        H_rad = H_pre_rad - (e * np.sinh(H_pre_rad) - H_pre_rad - (self.mu / -a**3)**0.5 *(t - t_p)) / (e * np.cosh(H_pre_rad) - 1)
        while(np.abs(H_pre_rad - H_pre_rad) >= self.threshold):
            H_pre_rad = H_pre_rad
            H_pre_rad = H_pre_rad - (e * np.sinh(H_pre_rad) - H_pre_rad - (self.mu / -a**3)**0.5 *(t - t_p)) / (e * np.cosh(H_pre_rad) - 1)
        return np.degrees(H_rad)
    
    def calc_PQW_from_orbital_elems(self,a,e,i,omega,Omega,t_p):
        """
        軌道要素からPQRベクトルを求める
        引数--------------------------
        a : double
            長半径(km)
        i : double
            軌道傾斜角(deg)
        e : double
            離心率
        omega : double
            近地点引数(deg)
        Omega : double
            昇交点離角(deg)
        t_p : double
            近地点通過時刻(s)
        返り値--------------------------
        P_hat_vec : ndarraty
            基準ベクトル
        Q_hat_vec : ndarraty
            基準ベクトル
        W_hat_vec : ndarraty
            基準ベクトル
        """
        omega_rad = np.radians(omega)
        Omega_rad = np.radians(Omega)
        i_rad = np.radians(i)
        #軌道面内単位ベクトル（近地点方向）
        P_hat_vec = np.array(
            [[np.cos(omega_rad) * np.cos(Omega_rad) - np.sin(omega_rad) * np.sin(Omega_rad) * np.cos(i_rad)],
             [np.cos(omega_rad) * np.sin(Omega_rad) + np.sin(omega_rad) * np.cos(Omega_rad) * np.cos(i_rad)],
             [np.sin(omega_rad) * np.sin(i_rad)]])
        #軌道面内単位ベクトル（近地点垂直方向）
        Q_hat_vec = np.array(
            [[-np.sin(omega_rad) * np.cos(Omega_rad) - np.cos(omega_rad) * np.sin(Omega_rad) * np.cos(i_rad)],
             [-np.sin(omega_rad) * np.sin(Omega_rad) + np.cos(omega_rad) * np.cos(Omega_rad) * np.cos(i_rad)],
             [np.cos(omega_rad) * np.sin(i_rad)]])     
        #軌道面垂直方向単位ベクトル
        W_hat_vec = np.array(
            [[np.sin(Omega_rad) * np.sin(i_rad)],
             [-np.cos(Omega_rad) * np.sin(i_rad)],
             [np.cos(i_rad)]])
        
        return P_hat_vec, Q_hat_vec, W_hat_vec
    
    def calc_r_v_form_r_v_0(self, r_vec0, v_vec0, JS0, JS):
        """
        calc (r,v) at JS from (r0,v0) at JS0
        input--------------------------
        r_vec0 : 3*1 ndarray(double)
            時刻JS0のrの値(km)
        v_vec0 : 3*1 ndarray(double)
            時刻JS0のvの値(km/s)
        JS0 : double
            initial time(s)
        JS : double
            time of returned r,v
        return--------------------------
        r_vec : 3*1 ndarray(double)
            時刻JSのrの値(km)
        v_vec : 3*1 ndarray(double)
            時刻JSのvの値(km/s)
        """
        a,e,i,omega,Omega,t_p,_,_,_ = self.calc_orbital_elems_from_r_v(self, r_vec0 ,v_vec0, JS0)
        return self.calc_r_v_from_orbital_elems(self,a,e,i,omega,Omega,t_p,JS)
    
    def calc_r_v_from_orbital_elems(self,a,e,i,omega,Omega,t_p,JS):
        """
        軌道要素からある時刻のr,vを求める
        引数--------------------------
        a : double
            長半径(km)
        i : double
            軌道傾斜角(deg)
        e : double
            離心率
        omega : double
            近地点引数(deg)
        Omega : double
            昇交点離角(deg)
        t_p : double
            近地点通過時刻(JS,s)
        JS : double
            r,vを求めたい時刻(s)
        返り値--------------------------
        r_vec : 3*1 ndarray(double)
            時刻JSのrの値(km)
        v_vec : 3*1 ndarray(double)
            時刻JSのvの値(km/s)
        """
        omega_rad = np.radians(omega)
        Omega_rad = np.radians(Omega)
        i_rad = np.radians(i)
        #軌道面内単位ベクトル（近地点方向）
        P_hat_vec, Q_hat_vec, _ =  self.calc_PQW_from_orbital_elems(a,e,i,omega,Omega,t_p)
        p = a * (1 - e**2)
        #楕円の場合
        if (a > 0):
            E = self.solve_Kepler(a,e,t_p,JS,100)
            E_rad = np.radians(E)
            r = a * (1 - e * np.cos(E_rad))
            r_vec = a * (np.cos(E_rad) - e) * P_hat_vec + (a * p)**0.5 * np.sin(E_rad) * Q_hat_vec
            v_vec = -(a * self.mu)**0.5 / r * np.sin(E_rad) * P_hat_vec + (self.mu * p)**0.5 / r * np.cos(E_rad) * Q_hat_vec
        #双曲線の場合（放物線は考えない）
        else:
            H = self.solve_hyperbola_eq(a,e,t_p,JS,100)
            H_rad = np.radians(H)
            r = a * (1 - e * np.cosh(H_rad))
            r_vec = a * (np.cosh(H_rad) - e) * P_hat_vec + (- a * p)**0.5 * np.sinh(H_rad) * Q_hat_vec
            v_vec = -(-a * self.mu)**0.5 / r * np.sinh(H_rad) * P_hat_vec + (self.mu * p)**0.5 / r * np.cosh(H_rad) * Q_hat_vec
        return r_vec, v_vec
    
    def calc_orbital_elems_from_r_v(self, r_vec ,v_vec, JS,
                                i_hat_vec = np.array([[1],[0],[0]]), j_hat_vec= np.array([[0],[1],[0]]), k_hat_vec= np.array([[0],[0],[1]])):
        """
        ある時刻のr,vから軌道要素を求める
        引数--------------------------
        r_vec : 3*1 ndarray(double)
            rの値(km)
        v_vec : 3*1 ndarray(double)
            vの値(km/s)
        JS : double
            時刻(s)
        i_hat_vec : 3*1 ndarray(double)
            赤道面の春分点方向基準ベクトル
        j_hat_vec : 3*1 ndarray(double)
            赤道面の春分点垂直方向基準ベクトル
        k_hat_vec : 3*1 ndarray(double)
            赤道面に垂直な基準ベクトル
        返り値--------------------------
        a : double
            長半径(km)
        i : double
            軌道傾斜角(deg)
        e : double
            離心率
        omega : double
            近地点引数(deg)
        Omega : double
            昇交点離角(deg)
        t_p : double
            近地点通過時刻(JS,s)
        P_hat_vec : ndarraty
            基準ベクトル
        W_hat_vec : ndarraty
            基準ベクトル
        Q_hat_vec : ndarraty
            基準ベクトル
        """
        r = np.linalg.norm(r_vec)
        epsilon = 0.5 * np.dot(v_vec.T,v_vec) - self.mu / r #エネルギー積分
        h_vec = np.cross(r_vec.T, v_vec.T).T #角運動量
        P_vec = np.cross(v_vec.T, h_vec.T).T - self.mu * r_vec / r #ラプラスベクトル
        P_hat_vec = P_vec / np.linalg.norm(P_vec) #基準ベクトル
        W_hat_vec = h_vec / np.linalg.norm(h_vec) #基準ベクトル
        Q_hat_vec = np.cross(W_hat_vec.T, P_hat_vec.T).T #基準ベクトル

        a = - self.mu / 2.0 / epsilon
        e = np.linalg.norm(P_vec) / self.mu
        i = np.degrees(np.arccos(np.dot(W_hat_vec.T,k_hat_vec)))
        Omega = np.degrees(np.arctan2( np.dot(W_hat_vec.T, i_hat_vec), - np.dot(W_hat_vec.T, j_hat_vec)))
        N_hat_vec = np.cross(k_hat_vec.T, h_vec.T).T / np.linalg.norm(np.cross(k_hat_vec.T, h_vec.T))  #昇交点方向基準ベクトル
        omega = np.degrees(np.arccos(np.dot(N_hat_vec.T, P_hat_vec)))
        if (np.dot(P_hat_vec.T, k_hat_vec) < 0):
            omega = 360 - omega
        if (a > 0): #楕円の場合
            E_rad  = np.arctan2(np.dot(r_vec.T, v_vec) / (self.mu * a)**0.5, (1 - r / a))
            t_p = JS - (a**3 / self.mu)**0.5 * (E_rad - e * np.sin(E_rad))
        else: #双曲線の場合
            H_rad  = np.arcsinh(np.dot(r_vec.T, v_vec) / (self.mu * -a)**0.5 / e)
            t_p = JS - (-a**3 / self.mu)**0.5 * (e * np.sin(H_rad) - H_rad)
        return float(a),float(e),float(i),float(omega),float(Omega),float(t_p),P_hat_vec, Q_hat_vec,W_hat_vec

    def eci2ecef(self, r, theta):
        """
        春分方向をx方向とする座標系から惑星固定座標系に
        引数--------------------------
        r : 3*1 ndarray(double)
            ECI座標系での座標
        theta : double
            グリニッジ恒星時(deg)
        返り値--------------------------
        r_ecef : 3*1 ndarray(double)
            ECEF座標系での座標
        """
        theta_rad = np.radians(theta)
        R = np.array([[np.cos(-theta_rad), -np.sin(-theta_rad), 0],[np.sin(-theta_rad), np.cos(-theta_rad), 0], [0, 0, 1]])
        r_ecef = np.dot(R, r)
        return r_ecef

    def calc_geodetic_position(self, r_ecef):
        """
        緯度・経度を求める
        引数--------------------------
        r_ecef : 3*1 ndarray(double)
            ECEF座標系での座標
        返り値--------------------------
        longitude : double
            経度
        latitude : double
            緯度
        """
        longitude_rad = np.arctan2(r_ecef[1], r_ecef[0])
        longitude = np.degrees(longitude_rad)

        r = np.sqrt(r_ecef[0]**2 + r_ecef[1]**2)
        latitude_rad = np.arctan2(r_ecef[2], r)
        latitude = np.degrees(latitude_rad)

        return longitude, latitude
    
class Planet():
    """
    惑星クラス、太陽周回軌道面や位置などの情報を与える
    """
    def __init__(self, planet_name, values = Values(), calculator  = TrajectoryCalculator):
        self.values = values
        self.planet_name = planet_name
        self.mu_sun = values.mu("Sun")
        self.calculator = calculator("Sun")
        self.mu = values.mu(planet_name)
        self.radius = values.radius(planet_name)
        self.rotation_period = values.rotation_period(planet_name)
        self.rotation_omega = 360 / self.rotation_period

    def position(self, year, month, date, hour, minute, second):
        """ある時刻の惑星のr,vを求める
        引数--------------------------
            時刻（太陽系力学時）
        返り値--------------------------
        r_vec : 3*1 ndarray(double)
            時刻tのrの値(km)
        v_vec : 3*1 ndarray(double)
            時刻tのvの値(km/s)
        """
        JS,JD,T_TDB = self.values.convert_times_to_T_TDB(year, month, date, hour, minute, second)
        a = self.values.a(self.planet_name, T_TDB) #長半径(km)
        e = self.values.e(self.planet_name, T_TDB) #離心率
        i = self.values.i(self.planet_name, T_TDB) #軌道面傾斜角(deg)
        Omega = self.values.Omega(self.planet_name, T_TDB) #昇交点経度(deg)
        omega = self.values.omega(self.planet_name, T_TDB) #近地点引数(deg)
        lambdaM = self.values.lambdaM(self.planet_name, T_TDB) #平均経度(deg)
        omega_tilde = omega + Omega
        M = lambdaM - omega_tilde
        M_rad = np.radians(M)
        t_p = JS - M_rad * (a**3/self.mu_sun)**0.5
        return self.calculator.calc_r_v_from_orbital_elems(a,e,i,omega,Omega,t_p,JS)
    
    def position_JS(self, JS):
        """ある時刻の惑星のr,vを求める
        引数--------------------------
            時刻（JD*24*60*60)
        返り値--------------------------
        r_vec : 3*1 ndarray(double)
            時刻tのrの値(km)
        v_vec : 3*1 ndarray(double)
            時刻tのvの値(km/s)
        """
        JD = JS / 24 / 60 / 60
        T_TDB = (JD - 2451545.0) / 36525.0
        a = self.values.a(self.planet_name, T_TDB) #長半径(km)
        e = self.values.e(self.planet_name, T_TDB) #離心率
        i = self.values.i(self.planet_name, T_TDB) #軌道面傾斜角(deg)
        Omega = self.values.Omega(self.planet_name, T_TDB) #昇交点経度(deg)
        omega = self.values.omega(self.planet_name, T_TDB) #近地点引数(deg)
        lambdaM = self.values.lambdaM(self.planet_name, T_TDB) #平均経度(deg)
        omega_tilde = omega + Omega
        M = lambdaM - omega_tilde
        M_rad = np.radians(M)
        t_p = JS - M_rad * (a**3/self.mu_sun)**0.5
        return self.calculator.calc_r_v_from_orbital_elems(a,e,i,omega,Omega,t_p,JS)
    
    def plot_trajectory(self,start_date=(2023,3,3,0,0,0),end_date=(2060,3,3,0,0,0)):
        start_JS = self.values.convert_times_to_T_TDB(*start_date)[0]
        end_JS = self.values.convert_times_to_T_TDB(*end_date)[0]
        x = np.array([])
        y = np.array([])
        for i in range(int(start_JS/100000),int(end_JS/100000)):
            r = self.position_JS(i*100000)[0]
            x = np.append(x,r[0])
            y = np.append(y,r[1])
        plt.plot(x,y,"k")