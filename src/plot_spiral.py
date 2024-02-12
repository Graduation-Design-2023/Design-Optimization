import numpy as np
from matplotlib import pyplot as plt
import json
# file_name = "outputs/runs/insertion2deploy.json"
file_name = "outputs/runs/insertion.json"

I_sp = 3000
g = 9.8 * 1e-32434

with open(file_name, "r") as f:
    result = json.load(f)
    result = list(result)
    r_his = np.array(result[0])
    m_his = np.array(result[1])
    dv = g * I_sp * np.log(m_his[0] / m_his[-1])
    print(m_his[0] - m_his[-1])
    print(m_his[0], m_his[-1])
    print(dv)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(r_his[0,:], r_his[1,:], r_his[2,:], linewidth=0.3)
ax.set_xlabel('x[km]')
ax.set_ylabel('y[km]')
ax.set_zlabel('z[km]')
# ax.set_title('trajectory from insertion to deploy')
ax.set_title('trajectory from deploy to observation orbit') 
plt.show()