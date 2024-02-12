from libs.lib import Values, TrajectoryCalculator, Planet
from libs.occultation_lib import Satellite, Occultation
from libs.interplanetary_lib import PlanetsTransOrbit

import numpy as np
from matplotlib import pyplot as plt

lat_ground, lon_ground = 31.2512709, 131.0761071
lat_ground_rad = np.deg2rad(lat_ground)
lon_ground_rad = np.deg2rad(lon_ground)
 #Values
myval = Values()
#Planet
earth = Planet("Earth")
mars = Planet("Mars")
#TrajectoryCalculator
earthalclator = TrajectoryCalculator("Earth")
R = earth.radius

r_ecef = np.array([[R * np.cos(lon_ground_rad) * np.cos(lat_ground_rad)],
                   [R * np.sin(lon_ground_rad) * np.cos(lat_ground_rad)], 
                   [R * np.sin(lat_ground_rad)]])

def path_time_1day(theta0, t0, dt):
    t_1day = 24 * 60 * 60
    num = int(t_1day / dt)
    theta = theta0
    t = t0
    t_list = []

    for i in range(num):
        r_eci = earthalclator.ecef2eci(r_ecef, theta)
        r_sun = earthalclator.eci2sun(r_eci, 23.5)
        r_e,_ = earth.position_JS(t)
        r_m,_ = mars.position_JS(t)
        r_rel = r_m - r_e
        if (np.dot(r_rel.T, r_sun) > 0):
            t_list.append(t)
        t += dt
        theta += dt * earth.rotation_omega
    return len(t_list) * dt


if __name__ == '__main__':
    t0,_,_ = myval.convert_times_to_T_TDB(2030,1,1,1,1,1)
    theta0 = 0 #シミュレーションスタート時の本初子午線からの角度
    num = int(365*1.4)
    dt = 50
    t_list = np.zeros(num)

    for i in range(num):
        t_list[i] = path_time_1day(theta0, t0, dt)
        t0 += 24 * 60 * 60
        theta0 += 24 * 60 * 60 * earth.rotation_omega

    plt.plot(np.arange(0,num,1), t_list/60)
    plt.title('visible time')
    plt.xlabel('time [day]')
    plt.ylabel('visible time [min]')
    plt.show()
