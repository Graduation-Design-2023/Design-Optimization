from libs.lib import Values, TrajectoryCalculator, Planet
from libs.occultation_lib import Satellite, Occultation
from libs.interplanetary_lib import PlanetsTransOrbit

import numpy as np
from matplotlib import pyplot as plt
import json
import argparse 

myval = Values()
earth = Planet("Earth")
mars = Planet("Mars")
jupiter = Planet("Jupiter")
#TrajectoryCalculator
earthalclator = TrajectoryCalculator("Earth")
sun_calculator = TrajectoryCalculator("Sun")
mars_calculator = TrajectoryCalculator("Mars")
jupiter_calculator = TrajectoryCalculator("Jupiter")

earth_mars = PlanetsTransOrbit(earth, mars)

#-------------------------------------------------
# ここを適切に書き換える（ホントはjsonに保存して読み出したい）
target_planet_list = [mars,earth,earth]
num_swingby = len(target_planet_list) - 2
result_file_name = 'outputs/runs/inter_planets_determined.json'
#-------------------------------------------------

# functions for converting X to arguments of trajectory_with_swingby
def append_X_scholar_to_list(list, num_swingby, X_i, j_start):
    for j in range(num_swingby):
        list.append(X_i[j_start+j])
    return j+j_start

def append_X_ndarray3_to_list(list, num_swingby, X_i, j_start):
    for j in range(num_swingby):
        vec = np.array([[X_i[j_start+3*j]], [X_i[j_start+3*j+1]], [X_i[j_start+3*j+2]]])
        list.append(vec)
    return j_start+3*j+2

if __name__ == '__main__':
    with open(result_file_name, "r") as f:
        results = json.load(f)
        result = results[0]

    print(result)
    j = 2
    arrival_JS_list = [] # num_swingby+2
    v_inf_in_list = [] # 3*(num_swingby+1)
    theta_list = [] # num_swingby
    h_list = [] # num_swingby
    JS_tcm_list = [] # num_swingby+1

    j = append_X_scholar_to_list(arrival_JS_list, num_swingby+2, result, j)
    j = append_X_ndarray3_to_list(v_inf_in_list, num_swingby+1, result, j+1)
    j = append_X_scholar_to_list(theta_list, num_swingby, result, j+1)
    j = append_X_scholar_to_list(h_list, num_swingby, result, j+1)
    j = append_X_scholar_to_list(JS_tcm_list, num_swingby+1, result, j+1)

    res = earth_mars.trajectory_with_swingby(target_planet_list, arrival_JS_list, v_inf_in_list, theta_list, h_list, JS_tcm_list, plot_is_enabled = "True")
    print('TCM: ', np.sum(res[0]))
    print('Swingby: ', np.sum(res[1]))
    print('v_inf_start: ', np.sum(res[2]))
    print('v_inf_end: ', np.sum(res[3]))