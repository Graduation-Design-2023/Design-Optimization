from libs.lib import Values, TrajectoryCalculator, Planet
import numpy as np
from matplotlib import pyplot as plt

earth = Planet("Earth")
mars = Planet("Mars")
myval = Values()
time_start,_,_ = myval.convert_times_to_T_TDB(2035,1,1,1,1,1)
time_end = time_start + 2*365*24*60*60
time_list = np.linspace(time_start, time_end, 100)
distance_list = np.zeros_like(time_list)
for i in range(len(time_list)):
    time = time_list[i]
    re, _ = earth.position_JS(time)
    rm , _= mars.position_JS(time)
    distance_list[i] = np.linalg.norm(re - rm)
print(distance_list)
plt.plot(distance_list)
plt.show()
