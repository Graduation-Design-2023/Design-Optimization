from libs.lib import Values, TrajectoryCalculator, Planet
import numpy as np
from matplotlib import pyplot as plot_trajectory

class Satellite():
    def __init__(self, planet, calculator = TrajectoryCalculator):
        self.planet = planet
        self.calculator = calculator(planet.planet_name)

    def init_orbit_by_orbital_elems(self, a, e, i, omega, Omega, t_p):
        self.orbital_elems = np.array([a, e, i, omega, Omega, t_p])

    def init_orbit_by_rv(self, r_vec ,v_vec, t0):
        self.orbital_elems = np.array([self.calculator.calc_orbital_elems_from_r_v(self, r_vec ,v_vec, t0)])
    def get_rv(self, t):
        r, v = self.calculator.calc_r_v_from_orbital_elems(*self.orbital_elems, t)
        return r, v

class Occultation():
    """
    衛星間電波遮蔽計算用のクラス
    """
    def __init__(self, planet, satellites, calculator  = TrajectoryCalculator):
        """ある時刻の惑星のr,vを求める
        引数--------------------------
            planet : Planet
                中心惑星
        """
        self.planet = planet
        self.sats = satellites
        self.trajectory_calculator = calculator(planet.planet_name)
        
    
    def get_position_observed(self, r1, r2):
        """観測点を求める
        引数--------------------------
            r1,r2 : 3*1 ndarray(double)
                2衛星のどちらかの位置ベクトル
        返り値--------------------------
        distance : double
            中心惑星と直線の距離
        r_h : 3*1 ndarray(double)
            垂線の脚の座標
        is_occultated : bool
            遮蔽されているか
        """
        r_rel = r2 - r1
        r_h = r1 - (np.dot(r1.T, r_rel) / np.linalg.norm(r_rel)**2 * r_rel)
        distance = np.linalg.norm(r_h)
        
        if (distance < self.planet.radius and np.dot((r1 - r_h).T,r2 - r_h) < 0):
            is_occultated = True
        else:
            is_occultated = False

        return distance, r_h, is_occultated

    def geodetic_position_observed(self, r_h, theta):
        """垂線の足の座標から電波遮蔽の観測点の緯度・経度を求める
        引数--------------------------
        r_h : 3*1 ndarray(double)
            垂線の脚の座標
        theta : double
            グリニッジ恒星時(deg)
        返り値--------------------------
        longitude : double
            観測点の経度
        latitude : double
            観測点の緯度
        """
        r_observed = r_h * self.planet.radius / np.linalg.norm(r_h)
        r_ecef = self.trajectory_calculator.eci2ecef(r_observed, theta)
        longitude, latitude = self.trajectory_calculator.calc_geodetic_position(r_ecef)
        return longitude, latitude

    def get_position_observed_mult(self, sats, t):
        """観測点を求める
        引数--------------------------
        sats : list(Satellite)
            衛星たち
        t : double
            時刻
        返り値--------------------------
        r_h_list : n*n*3 ndarray(double)
            垂線の脚の座標のリスト
        is_occultated_list : n*n list(bool)
            遮蔽されているかのリスト
        """
        n = len(sats)
        is_occultated_list = np.full((n,n),False)
        r_h_list = np.zeros((n,n,3))
        for i in range(n):
            for j in range(n):
                if (i == j):
                    break
                ri, _ = sats[i].get_rv(t)
                rj, _ = sats[j].get_rv(t)
                _, r_h, is_occultated = self.get_position_observed(ri, rj)
                r_h_list[i][j] = r_h.T
                is_occultated_list[i][j] = is_occultated
        return r_h_list, is_occultated_list

    def simulate_position_observed(self, theta0, t0, t_end, dt):
        """シミュレーションにより観測点を求める
        引数--------------------------
        theta0 :double
            t0でのグリニッジ恒星時(deg)
        t0 : double
            初期時刻
        t_end : double
            シミュレーション終了時刻
        dt : double
            シミュレーションステップ幅
        返り値--------------------------
        latitude_list : ndarray(double)
            観測点のlatitude
        longitude_list : ndarray(double)
            観測点のlongitude
        """
        n_step = int((t_end - t0) / dt)
        n = len(self.sats)

        is_occultated_list = np.full((n,n),False)
        prev_is_occultated_list = np.full((n,n),False)
        is_first_list =  np.full((n,n),True)
        mask_tri = np.tril(np.full((n,n),True), k = -1) # 上半分と下半分でsat1→sat2とsat2→sat1が重複するため、片方だけ取ってくる
        r_h_list = np.zeros((n,n,3))
        latitude_list = np.array([])
        longitude_list = np.array([])
        count = 0

        t = t0
        theta = theta0
        for i in range(n_step):
            r_h_list, is_occultated_list = self.get_position_observed_mult(self.sats, t)

            # 掩蔽開始時
            mask1 = is_occultated_list & is_first_list & mask_tri
            r_h_true_list1 = r_h_list[mask1]
            for j in range(len(r_h_true_list1)):
                count += 1
                longitude, latitude =  self.geodetic_position_observed(r_h_true_list1[j], theta)
                latitude_list = np.append(latitude_list, latitude)
                longitude_list = np.append(longitude_list, longitude)

            # 掩蔽終了時
            mask2 = ~is_occultated_list & prev_is_occultated_list & mask_tri
            r_h_true_list2 = r_h_list[mask2]
            for j in range(len(r_h_true_list2)):
                count += 1
                longitude, latitude =  self.geodetic_position_observed(r_h_true_list2[j], theta)
                latitude_list = np.append(latitude_list, latitude)
                longitude_list = np.append(longitude_list, longitude)

            prev_is_occultated_list[is_occultated_list] = True
            prev_is_occultated_list[~is_occultated_list] = False
            is_first_list[is_occultated_list] = False
            is_first_list[~is_occultated_list] = True
            
            t += dt
            theta += dt * self.planet.rotation_omega

        return longitude_list, latitude_list, count