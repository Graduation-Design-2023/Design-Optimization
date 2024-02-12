from libs.lib import Values, TrajectoryCalculator, Planet
from libs.occultation_lib import Satellite, Occultation
from libs.interplanetary_lib import PlanetsTransOrbit

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize # Normalizeをimport
import json
import argparse

#Values
myval = Values()
#Planet
earth = Planet("Earth")
mars = Planet("Mars")
#TrajectoryCalculator
earthalclator = TrajectoryCalculator("Earth")
sun_calculator = TrajectoryCalculator("Sun")
mars_calculator = TrajectoryCalculator("Mars")

def calc_eclipse_time_from_1angle(sat, sun_angle):
    a = sat.orbital_elems[0]
    n = (sat.planet.mu / a**3)**0.5
    sun_angle_rad = np.deg2rad(sun_angle)
    sun_direction = np.array([[np.cos(sun_angle_rad)], [np.sin(sun_angle_rad)], [0]])

    t_end = 2 * np.pi / n
    dt = 1
    num = int(t_end /dt)
    is_eclipsed_list = np.full(num, False)

    for i in range(num):
        t = i * dt
        r_vec, _ = sat.get_rv(t)
        r_parallel = np.dot(sun_direction.T, r_vec)
        if (r_parallel > 0):
            r_parallel_vec = r_parallel * sun_direction
            r_vertial = np.linalg.norm(r_vec - r_parallel_vec)
            if (r_vertial < sat.planet.radius):
                is_eclipsed_list[i] = True
    elipse_time = np.count_nonzero(is_eclipsed_list) * dt
    return elipse_time

def simulate_elipse_time(sat, title):
    T_TDB = myval.convert_times_to_T_TDB(2030,1,1,1,1,1)
    a = myval.a(sat.planet.planet_name, T_TDB[0])
    n = (myval.mu('Sun') / a**3)**0.5

    dtheta = 5
    num = int(360 / dtheta)
    sun_angle_list = np.arange(0, 360, dtheta)
    elipsed_time_list = np.zeros(num)

    for i in range(num):
        elipsed_time_list[i] = calc_eclipse_time_from_1angle(sat, sun_angle_list[i])
    time_list = np.deg2rad(np.arange(0, 360, dtheta)) / n / 60 / 60 / 24 
    elipsed_time_list /= 60 * 60

    plt.plot(time_list, elipsed_time_list)
    plt.title(title)
    plt.xlabel('time [days]')
    plt.ylabel('elipse time [hour]')
    plt.show()


sat1 = Satellite(mars)
sat2 = Satellite(mars)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("result_file_name")
    # parser.add_argument("selected_pareto_idx", type=int)
    # args = parser.parse_args()
    # with open(args.result_file_name, "r") as f:
    #     results = json.load(f)
    #     result = results[args.selected_pareto_idx]

    result_file_name = "outputs/runs/obs.json"
    selected_pareto_idx = 0
    with open(result_file_name, "r") as f:
        results = json.load(f)
        result = results[selected_pareto_idx]

    X  = result[1:]
    print(result)
    i = 0
    oe_observation1 = (X[0], X[1], X[2], X[3], X[4], X[5])
    oe_observation2 = (X[6], X[7], X[8], X[9], X[10], X[11])
    sat1.init_orbit_by_orbital_elems(*oe_observation1)
    sat2.init_orbit_by_orbital_elems(*oe_observation2)
    simulate_elipse_time(sat1, "elipse time of a chief sat")
    simulate_elipse_time(sat2, "elipse time of a deputy sat")