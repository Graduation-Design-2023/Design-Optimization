from libs.lib import Values, TrajectoryCalculator, Planet, Satellite, Occultation, PlanetsTransOrbit

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
    res = earth_mars.calc_launch_window(2024, 4, 1, 0.001, 10)
    print(res)