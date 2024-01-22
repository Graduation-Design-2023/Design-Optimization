from libs.lib import Values, TrajectoryCalculator, Planet
from libs.occultation_lib import Satellite, Occultation
from libs.interplanetary_lib import PlanetsTransOrbit

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    myval = Values()
    earth = Planet("Earth")
    mars = Planet("Mars")
    jupiter = Planet("Jupiter")
    #TrajectoryCalculator
    earthalclator = TrajectoryCalculator("Earth")
    sun_calculator = TrajectoryCalculator("Sun")
    mars_calculator = TrajectoryCalculator("Mars")
    jupiter_calculator = TrajectoryCalculator("Jupiter")

    earth_jupiter = PlanetsTransOrbit(earth, jupiter)
    target_planet_list = [mars]

    JS_launch, _, _ = myval.convert_times_to_T_TDB(1984, 2, 20, 0, 0, 0)
    JS_swingby, _, _ = myval.convert_times_to_T_TDB(1984, 5, 29, 0, 0, 0)
    JS_end, _, _ = myval.convert_times_to_T_TDB(1986, 5, 17, 0, 0, 0)
    arrival_JS_list = [JS_end, JS_swingby, JS_launch]

    v_inf_0 = np.array([[2],[-5],[0.1]])
    v_inf_1 = np.array([[5],[-0],[0]])
    v_inf_list = [v_inf_0, v_inf_1]

    theta_list = [11.3]
    h_list = [200]
    JS_tcm, _, _ = myval.convert_times_to_T_TDB(1984, 6, 1, 0, 0, 0)
    JS_tcm_list = [JS_tcm, 0.5 * (JS_launch + JS_swingby)]

    res = earth_jupiter.trajectory_with_swingby(target_planet_list, arrival_JS_list, v_inf_list, theta_list, h_list, JS_tcm_list, plot_is_enabled = "True")
    print(res)