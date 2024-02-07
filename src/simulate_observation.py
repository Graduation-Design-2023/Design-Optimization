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
    t_end = 3 * 24 * 60**2
    dt = 500

    a_c = 9672.519052858057
    e_c = 0.38908928286671307
    i_c = 39.852070553188724
    omega_c = 58.948281601252035
    Omega_c = 340.8771431050665
    tp_c = 4623.607735280737

    a_d1 = 3450.554541372235
    e_d1 = 0.00398799817823215
    i_d1 = 36.008088428304944
    omega_d1 = 56.394733700507274
    Omega_d1 = 334.1376363351017
    tp_d1 = 15314.074981002512

    sat1 = Satellite(target_planet)
    sat2 = Satellite(target_planet)
    
    sat1.init_orbit_by_orbital_elems(a_c, e_c, i_c, omega_c, Omega_c, tp_c)
    sat2.init_orbit_by_orbital_elems(a_d1, e_d1, i_d1, omega_d1, Omega_d1, tp_d1)
    
    #OccultationCalculator
    venus_occultation = Occultation(target_planet,[sat1,sat2])
    longitude_list, latitude_list, count = venus_occultation.simulate_position_observed(0, t0, t_end, dt)
    plt.plot(longitude_list, latitude_list,'.')
    plt.show()
    plt.savefig('z.png')