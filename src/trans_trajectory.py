from libs.lib import Values, TrajectoryCalculator, Planet,PlanetsTransOrbit, Satellite, Occultation

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

    num_window = 10
    search_width = 40*24*60*60
    num_step = 100
    step = search_width / num_step

    v_inf_start_array = np.zeros((num_window,num_step))
    v_inf_end_array = np.zeros((num_window,num_step))

    windows , t_H = earth_mars.calc_launch_window(2024, 4, 1, 0.001, num_window)
    duration = t_H - search_width/2

    for i in range(num_window):
        for count in range(num_step):
            _,_,_,_,v_planet_start,v_planet_end,v_sat_start,v_sat_end = earth_mars.trajectory_by_lambert(windows[i], duration)
            v_inf_start_array[i][count] = np.linalg.norm(v_planet_start - v_sat_start)
            v_inf_end_array[i][count] = np.linalg.norm(v_planet_end - v_sat_end)
            duration += step
        plt.plot(v_inf_end_array[i], v_inf_start_array[i],label=str(windows[i][0:3])) 
        plt.xlabel("v_inf_end(km/s)")
        plt.ylabel("v_inf_start(km/s)")
        plt.xlim(0,10)
        plt.ylim(0,10)
    plt.legend() 
    plt.show()

    