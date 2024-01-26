from libs.lib import Values, TrajectoryCalculator, Planet
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_bvp
#Values
myval = Values()
#Planet
earth = Planet("Earth")
mars = Planet("Mars")

# 初期条件
start = (1971, 10, 2, 0, 0, 0)
tf = 184 * 24 * 60 * 60
js_start, _, _ = myval.convert_times_to_T_TDB(*start)
js_end = js_start + tf
mu = 1.32712440 * 10**11


r0, v0 = earth.position_JS(js_start)
rf, vf = mars.position_JS(js_end)

x0, y0, z0 = r0[:, 0]
vx0, vy0, vz0 = v0[:, 0]

xf, yf, zf = rf[:, 0]
vxf, vyf, vzf = vf[:, 0]

def bvpfun(x, y):
    aT_r = y[6] * y[0] + y[7] * y[1] + y[8] * y[2]
    r = np.sqrt(y[0]**2 + y[1]**2 + y[2]**2)

    dydx = np.array([y[3],
                    y[4],
                    y[5],
                    y[6] - mu * y[0] / r**3,
                    y[7] - mu * y[1] / r**3,
                    y[8] - mu * y[2] / r**3,
                    y[9],
                    y[10],
                    y[11],
                    -mu / r**3 * y[6] + 3 * mu / r**5 * aT_r * y[0],
                    -mu / r**3 * y[7] + 3 * mu / r**5 * aT_r * y[1],
                    -mu / r**3 * y[8] + 3 * mu / r**5 * aT_r * y[2]])
    return dydx

def bcfun(ya, yb):
    res = np.array([ya[0] - x0,
                    ya[1] - y0,
                    ya[2] - z0,
                    ya[3] - vx0,
                    ya[4] - vy0,
                    ya[5] - vz0,
                    yb[0] - xf,
                    yb[1] - yf,
                    yb[2] - zf,
                    yb[3] - vxf,
                    yb[4] - vyf,
                    yb[5] - vzf])
    return res

def guess(x):
    y = np.array([(xf - x0) / tf * xi + x0,
                  (yf - y0) / tf * xi + y0,
                  (zf - z0) / tf * xi + z0,
                  (vxf - vx0) / tf * xi + vx0,
                  (vyf - vy0) / tf * xi + vy0,
                  (vzf - vz0) / tf * xi + vz0,
                  0,
                  0,
                  0,
                  0,
                  0,
                  0])
    return y

# 数値解の計算
xmesh = np.linspace(0, tf, 1000)
solinit = np.zeros((12, xmesh.size))
for i, xi in enumerate(xmesh):
    solinit[:, i] = guess(xi)

sol = solve_bvp(bvpfun, bcfun, xmesh, solinit)

j = sum(sol.y[6]**2 + sol.y[7]**2 + sol.y[8]**2) * xmesh[1]
print(j)
# 解のプロット
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(sol.y[0], sol.y[1], sol.y[2], label='Trajectory')
ax.scatter([x0], [y0], [z0], color='red', label='Start')
ax.scatter([xf], [yf], [zf], color='green', label='End')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
