import sys
sys.path.append("Design-Optimization")
from libs.lib import Values, TrajectoryCalculator, Planet, Satellite, Occultation

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
    t_end = 1 * 24 * 60**2
    dt = 500

    a_c = target_planet.radius + 2124
    e_c = 0.2231
    i_c = 0
    omega_c = 50
    Omega_c = 0
    tp_c = 0

    a_d1 = target_planet.radius + 37124
    e_d1 = 0.8529
    i_d1 = 20
    omega_d1 = 50
    Omega_d1 = 0
    tp_d1 = 0
    tp_d2 = 2 * np.pi / (target_planet.mu / a_d1**3)**0.5 * 105 / 360

    sat1 = Satellite(target_planet)
    sat2 = Satellite(target_planet)
    sat3 = Satellite(target_planet)
    sat1.init_orbit_by_orbital_elems(a_c, e_c, i_c, omega_c, Omega_c, tp_c)
    sat2.init_orbit_by_orbital_elems(a_d1, e_d1, i_d1, omega_d1, Omega_d1, tp_d1)
    sat3.init_orbit_by_orbital_elems(a_d1, e_d1, i_d1, omega_d1, Omega_d1, tp_d2)
    #OccultationCalculator
    venus_occultation = Occultation(target_planet,[sat1,sat2])
    longitude_list, latitude_list, count = venus_occultation.simulate_position_observed(0, t0, t_end, dt)
    plt.plot(longitude_list, latitude_list,'.')
    plt.show()