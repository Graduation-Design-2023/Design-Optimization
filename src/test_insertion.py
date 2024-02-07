from libs.lib import Values, TrajectoryCalculator, Planet
from libs.occultation_lib import Satellite, Occultation
from libs.interplanetary_lib import PlanetsTransOrbit

import numpy as np
from matplotlib import pyplot as plt
import json
import argparse 

#Values
myval = Values()
#Planet
mercury = Planet("Mercury")
venus = Planet("Venus")
earth = Planet("Earth")
mars = Planet("Mars")
jupiter = Planet("Jupiter")
uranus = Planet("Uranus")
neptune = Planet("Neptune")
#TrajectoryCalculator
earthalclator = TrajectoryCalculator("Earth")
sun_calculator = TrajectoryCalculator("Sun")
mars_calculator = TrajectoryCalculator("Mars")
venus_calculator = TrajectoryCalculator("Venus")
earth_mars = PlanetsTransOrbit(earth, mars)
# cange here-------------------------------------
windows , t_H = earth_mars.calc_launch_window(2024, 4, 1, 0.001, 1)
JS_launch, _, _ = myval.convert_times_to_T_TDB(2024, 4, 1, 0, 0, 0)
duration = t_H
JS0 = JS_launch + duration
v_inf_vec = np.array([[0.4],[0.1],[0.1]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file_name")
    parser.add_argument("selected_pareto_idx", type=int)
    args = parser.parse_args()
    with open(args.result_file_name, "r") as f:
        results = json.load(f)
        result = results[args.selected_pareto_idx]
        
    
    oe_observation1 = (result[5], result[6], result[7], result[8], result[9], result[10])
    oe_observation2 = (result[11], result[12], result[13], result[14], result[15], result[16])
    theta, r_h, r_a = result[2], result[3], result[4]

    target_is_perigee = True
    val = earth_mars.trajectory_insertion_2sats(theta, r_h, v_inf_vec, JS0, r_a, oe_observation1,oe_observation2,target_is_perigee)
    print(val)