import numpy as np
from matplotlib import pyplot as plt
import json
# file_name = "outputs/runs/insertion2deploy.json"
file_name = "outputs/runs/deploy2obs.json"
with open(file_name, "r") as f:
    result_ = json.load(f)
    result_ = list(result_)[0]
    result = np.array(result_)
    print(result.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(result[0,:], result[1,:], result[2,:], linewidth=0.3)
ax.set_xlabel('x[km]')
ax.set_ylabel('y[km]')
ax.set_zlabel('z[km]')
# ax.set_title('trajectory from insertion to deploy')
ax.set_title('trajectory from deploy to observation orbit') 
plt.show()