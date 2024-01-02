from libs.lib import Values, TrajectoryCalculator, Planet
from libs.occultation_lib import Satellite, Occultation
from libs.interplanetary_lib import PlanetsTransOrbit

import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
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
    a = mars.radius + 21240
    e = 0.2231
    i = 80
    omega = 50
    Omega = 0
    tp= 0 #invalid
    oe_observation = (a, e, i, omega, Omega, tp)

    theta = 30
    r_h = mars.radius * 2

    windows , t_H = earth_mars.calc_launch_window(2024, 4, 1, 0.001, 1)
    JS_launch, _, _ = myval.convert_times_to_T_TDB(2024, 4, 1, 0, 0, 0)
    duration = t_H

    _,_,_,_,v_planet_start,v_planet_end,v_sat_start,v_sat_end = earth_mars.trajectory_by_lambert(windows[0], duration)
    v_inf_vec = v_sat_end - v_planet_end

    JS = JS_launch + duration
    target_is_perigee = False
    earth_mars.trajectory_insertion(theta, r_h, v_inf_vec, JS, 5*r_h, oe_observation,target_is_perigee)