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

#----------------------------------------
# シミュレーション条件に合わせて変更すること
theta0 = 0 
target_planet = mars
t0 = 0
t_end = 3 * 31 * 24 * 60**2
dt = 50
#-----------------------------------------

n_step = int((t_end - t0) / dt)
sat1 = Satellite(target_planet)
sat2 = Satellite(target_planet)

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
    #OccultationCalculator
    occultation = Occultation(mars,[sat1,sat2])

    ret, min_span_list = occultation.check_requirement_is_satisfied(theta0)
    min_span_list = np.where(min_span_list <= 0, 100, min_span_list)
    fig, ax = plt.subplots()
    shape = min_span_list.shape
    x_cs = np.array(range(shape[0])) * 360 / shape[0] - 180
    y_cs = np.array(range(shape[1])) * 180 / shape[1] - 90
    im = ax.imshow(min_span_list.T / 60 / 60, cmap='Greys', extent=[-180, 180, -90, 90], origin='lower', vmin=0, vmax=6)
    cbar = fig.colorbar(im)
    im.cmap.set_under('black')
    ax.set_title('observation interval [hour]')
    ax.set_xlabel('longitude [deg]')
    ax.set_ylabel('latitude [deg]')

    plt.show()

    if (ret == False):
        exit
    longitude_list, latitude_list, r1_list, r2_list, time_list = occultation.simulate_position_observed_2sats_detailed(0, t0, t_end, dt)

    plt.plot(longitude_list, latitude_list, '.') 
    plt.title('observed points ' + str(int(t_end/60/60/24))+ '[days]')
    plt.xlabel('longitude[deg]')
    plt.ylabel('latitude[deg]')
    plt.grid(which='major', axis='both', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-180, 180, 30)) 
    plt.yticks(np.arange(-90, 90, 30)) 
    plt.legend()
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r1_list[0,:], r1_list[1,:], r1_list[2,:])
    ax.plot(r2_list[0,:], r2_list[1,:], r2_list[2,:])
    ax.plot(0,0,0,'.',markersize=10)
    ax.legend(['sat1', 'sat2'])
    ax.set_xlabel('x[km]')
    ax.set_ylabel('y[km]')
    ax.set_zlabel('z[km]')
    plt.show()

    t_end=6*60*60
    dt = 0.1
    _, _, _, _, time_list, count = occultation.simulate_schedule_2sats(0, t0, t_end=t_end, dt = dt)
    print(count)
    time_list = np.array(time_list) / 60 / 60
    array_temp = np.zeros_like(time_list)
    plt.plot(time_list,array_temp, '.')
    plt.title('observation schedule' + str(t_end/60/60)+ '[hour]')
    plt.xlabel("time[hour]")
    plt.show()
    print(dt * len(time_list) / count)
