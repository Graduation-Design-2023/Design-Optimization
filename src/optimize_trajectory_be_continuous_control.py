from libs.lib import Values, TrajectoryCalculator, Planet

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_bvp
import os
import json
from datetime import datetime

from pymoo.util.misc import stack
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

#Values
myval = Values()
#Planet
earth = Planet("Earth")
mars = Planet("Mars")

mu = 1.32712440 * 10**11

# ----------------------------
# X : [js_start, tf]
t_launch0 = (2030, 1, 1, 0, 0, 0)
t_launch1 = (2040, 1, 1, 0, 0, 0)
js_launch0, _, _ = myval.convert_times_to_T_TDB(*t_launch0)
js_launch1, _, _ = myval.convert_times_to_T_TDB(*t_launch1)

tf0 = 200 * 24 * 60 * 60
tf1 = 400 * 24 * 60 * 60

xl_array = np.array([js_launch0, tf0])
xu_array = np.array([js_launch1, tf1])

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=0,
                         xl=xl_array,
                         xu=xu_array,
                        )
        self.my_bvp = Bvp()
        
    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X[:,0])
        f1 = np.zeros(n)
        f2 = np.zeros(n)
        for i in range(0, n):
            self.my_bvp.set_vals(X[i,0], X[i,1])
            f1[i] = self.my_bvp.solve(plot_is_enabled=False)
            f2[i] = X[i,1]

        out["F"] = np.column_stack([f1, f2])


class Bvp():
    def set_vals(self, js_start, tf):
        self.r0, self.v0 = earth.position_JS(js_start)
        self.rf, self.vf = mars.position_JS(js_start + tf)
        self.tf = tf
        self.js_start = js_start
        self.js_end = js_start + tf

    def bvpfun(self, x, y):
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

    def bcfun(self, ya, yb):
        x0, y0, z0 = self.r0[:, 0]
        vx0, vy0, vz0 = self.v0[:, 0]
        xf, yf, zf = self.rf[:, 0]
        vxf, vyf, vzf = self.vf[:, 0]

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

    def guess_by_planet_pos(self, t):
        re, ve = earth.position_JS(self.js_start + t)
        rm, vm = mars.position_JS(self.js_start + t)
        weight = t / self.tf
        y = np.array([(1 - weight) * re[0][0] + weight * rm[0][0],
                    (1 - weight) * re[1][0] + weight * rm[1][0],
                    (1 - weight) * re[2][0] + weight * rm[2][0],
                    (1 - weight) * ve[0][0] + weight * vm[0][0],
                    (1 - weight) * ve[1][0] + weight * vm[1][0],
                    (1 - weight) * ve[2][0] + weight * vm[2][0],
                    0,
                    0,
                    0,
                    0,
                    0,
                    0])
        return y

    def guess_linear(self, x):
        x0, y0, z0 = self.r0[:, 0]
        vx0, vy0, vz0 = self.v0[:, 0]
        xf, yf, zf = self.rf[:, 0]
        vxf, vyf, vzf = self.vf[:, 0]

        y = np.array([(xf - x0) / self.tf * x + x0,
                    (yf - y0) / self.tf * x + y0,
                    (zf - z0) / self.tf * x + z0,
                    (vxf - vx0) / self.tf * x + vx0,
                    (vyf - vy0) / self.tf * x + vy0,
                    (vzf - vz0) / self.tf * x + vz0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0])
        return y
    
    def solve(self, plot_is_enabled = True):
        """
        return : km^2/s^3
        """
        xmesh = np.linspace(0, self.tf, 1000)
        solinit = np.zeros((12, xmesh.size))
        for i, xi in enumerate(xmesh):
            solinit[:, i] = self.guess_by_planet_pos(xi)

        sol = solve_bvp(self.bvpfun, self.bcfun, xmesh, solinit)

        j = sum((sol.y[6]**2 + sol.y[7]**2 + sol.y[8]**2)**0.5) * xmesh[1]
        # print(j)
        if (plot_is_enabled):
            plt.plot(sol.y[0], sol.y[1], label='Trajectory')
            # plt.plot(solinit[0], solinit[1], label='Trajectory_Guess')s
            x0, y0, z0 = self.r0[:, 0]
            xf, yf, zf = self.rf[:, 0]
            plt.plot([x0], [y0],'.', color='red', label='Earth')
            plt.plot([xf], [yf],'.', color='green', label='Mars')
            plt.legend()
            plt.show()
            plt.savefig('outputs/runs/a.png')
        return j

if __name__ == '__main__':
    now = datetime.now()
    folder_name = "outputs/runs"
    file_name = f"{folder_name}/{now.strftime('%Y%m%d%H%M%S')}.json"
    os.makedirs(folder_name, exist_ok=True)

    problem = MyProblem()
    
    algorithm = NSGA2(
        pop_size=10,
        n_offsprings=10,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20),
        # eliminate_duplicates=True
    )
    termination = get_termination("n_gen", 20)
    
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    
    ps = problem.pareto_set(use_cache=False, flatten=False)
    pf = problem.pareto_front(use_cache=False, flatten=False)
    pop = res.pop
    
    # 目的関数空間
    plot = Scatter(title = "Objective Space", labels = ['dV [km/s]', 'tf [day]'])
    # res.F[:,0] *= 10**6 # km^2 -> m^2
    res.F[:,1] /= 24 * 60 * 60 # sec -> day
    plot.add(res.F, color = "red")

    output = np.concatenate([res.F, res.X], axis=1)
    print(output)
    with open(file_name, "w") as f:
        json.dump(output.tolist(), f, indent=" ")

    plot.show()
    # plot.save('outputs/runs/objective_space.png')

    X  = res.X
    i = 0
    my_bvp = Bvp()
    my_bvp.set_vals(X[i,0], X[i,1])
    print(my_bvp.solve(plot_is_enabled = True))
    