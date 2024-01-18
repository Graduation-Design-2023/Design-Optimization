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

    theta0 = 0 #fix me
    target_planet = mars
    t0 = 0
    t_end = 30 * 24 * 60**2
    dt = 60
    import json
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file_name")
    parser.add_argument("selected_pareto_idx", type=int)
    args = parser.parse_args()
    with open(args.result_file_name, "r") as f:
        results = json.load(f)
        result = results[args.selected_pareto_idx]
        num_sat = (len(result) - 3) // 6
        print(num_sat)
    sat_list = []

    for i in range(num_sat):
        sat = Satellite(target_planet)
        a, e, i, omega, Omega, tp = result[5 + 6 * i: 5 + 6 * (i + 1)]
        # a, e, i, omega, Omega, tp = result[3 + 6 * i: 3 + 6 * (i + 1)]
        sat.init_orbit_by_orbital_elems(a, e, i, omega, Omega, tp)
        sat_list.append(sat)
    
    #OccultationCalculator
    mars_occultation = Occultation(target_planet, sat_list)
    longitude_list, latitude_list, count = mars_occultation.simulate_position_observed(0, t0, t_end, dt)
    plt.plot(longitude_list, latitude_list,'.')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    xticks=np.arange(-180, 180, 30)
    yticks=np.arange(-90, 90, 30)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid()
    plt.show()
    plt.savefig("outputs/simulate_observation.png")