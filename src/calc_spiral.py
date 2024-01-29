from libs.lib import Values, TrajectoryCalculator, Planet
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
#Values
myval = Values()
#Planet
earth = Planet("Earth")
mars = Planet("Mars")

mu = mars.mu
mass = 100
a_theta = -10 * 10**(-3) / mass # はやぶさ 10 mN
# a_theta = -45 * 10**(-3) / mass # Masmi 45 mN
# a_theta = -89 * 10**(-3) / mass 

def ivpfun(t, y):
    dydt = np.array([y[1],
                     (y[2]**2) / y[0]**3 - mu / y[0]**2,
                     y[0] * a_theta])
    if (y[2] < (mu * rp)**0.5):
        dydt[2] = 0
    return dydt

v_inf = 0
rp = mars.radius*1.5
vp = (v_inf**2 + 2 * mu / rp)**0.5
h0 = rp*vp
y0 = [rp, 0, h0]
tf = 1 * 24 * 60 * 60

fig,ax = plt.subplots()

span = (0, tf)
solver = solve_ivp(ivpfun, span, y0, t_eval=np.linspace(*span,1000))
t = solver.t
y = solver.y
ax.plot(t, y[2])

plt.show()
plt.savefig('h.png')