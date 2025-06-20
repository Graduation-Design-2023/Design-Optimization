import numpy as np
from scipy.optimize import fsolve

def func_maximum_payload(payload_mass: float, Isp_list: list, M_i_list: list, M_f_list: list, C3: float, radius_parking=0.0) -> float:
    """function to solve maximum payload

    Args:
        payload_mass (float): payload mass
        Isp_list (list): 第i段の比推力
        M_i_list (list): 第i段の初期全備質量
        M_f_list (list): 第i段の周期全備質量
        radius_parking (float): パーキング軌道の地心距離
        C3 (float): 打ち上げエネルギー
    Returns:
        float: f(payload_mass)
    """
    g_0 = 9.81e-3  # 地上の重力加速度 [km / s^2]
    mu = 3.986e5  # 地球の重力定数 [km^3 / s^2]
    v_loss = 1.524 * 1.1  # 速度損失（経験値）[km/s]

    v = np.sqrt(C3 + 2 * mu / radius_parking)
    output = v + v_loss
    for Isp, M_i, M_f in zip(Isp_list, M_i_list, M_f_list):
        output -= g_0 * Isp * np.log((M_i + payload_mass) / (M_f + payload_mass))
    return output


def solve_maximum_payload(Isp_list, M_i_list, M_f_list, C3, radius_parking=250):
    # 初期推定値を設定
    initial_guess = 0.1  # [ton]

    # fsolve関数を使用して方程式を解く
    result = fsolve(func_maximum_payload, initial_guess, args=(Isp_list, M_i_list, M_f_list, C3, radius_parking))
    return result


