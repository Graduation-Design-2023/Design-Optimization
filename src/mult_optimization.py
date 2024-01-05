# reference : https://yuyumoyuyu.com/2021/07/23/howtousepymoo/

from libs.lib import Values, TrajectoryCalculator, Planet
from libs.occultation_lib import Satellite, Occultation
from libs.interplanetary_lib import PlanetsTransOrbit

import numpy as np
from matplotlib import pyplot as plt

from pymoo.util.misc import stack
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# Values
myval = Values()
# Planet
earth = Planet("Earth")
mars = Planet("Mars")
# TrajectoryCalculator
earth_calculator = TrajectoryCalculator("Earth")
sun_calculator = TrajectoryCalculator("Sun")
mars_calculator = TrajectoryCalculator("Mars")
# PlanetTrans
earth_mars = PlanetsTransOrbit(earth, mars)
sat1 = Satellite(mars)
sat2 = Satellite(mars)
#vals
t0 = 0
t_end = 1 * 24 * 60**2
dt = 500
target_is_perigee = True
radius = mars.radius
period = 2 * np.pi / (mars.mu / (radius*2)**3)**0.5

# solve lambert from earth to mars
windows , t_H = earth_mars.calc_launch_window(2024, 4, 1, 0.001, 1)
JS_launch, _, _ = myval.convert_times_to_T_TDB(2024, 4, 1, 0, 0, 0)
duration = t_H
JS0_in = JS_launch + duration
_,_,_,_,v_planet_start,v_planet_end,v_sat_start,v_sat_end = earth_mars.trajectory_by_lambert(windows[0], duration)
v_inf_vec = v_sat_end - v_planet_end

xl_array = np.array([0, radius, radius, radius, 0, 0, 0, 0, 0, radius, 0, 0, 0, 0, 0])
xu_array = np.array([360, radius*2, radius*3, radius*1.5, 1, 180, 360, 360, period, radius*1.5, 1, 180, 360, 360, period])
# ----------------------------
# X : [theta, r_h, r_a, a1, e1, i1, omega1, Omega1, t_p1, a2, e2, i2, omega2, Omega2, t_p2]
class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=15,
                         n_obj=2,
                         n_constr=0,
                         xl=xl_array,
                         xu=xu_array,
                        )
        
    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X[:,0])
        f1 = np.zeros(n)
        f2 = np.zeros(n)
        f3 = np.zeros(n)
        for i in range(0, n):
            oe_observation1 = (X[i,3], X[i,4], X[i,5], X[i,6], X[i,7], X[i,8])
            oe_observation2 = (X[i,9], X[i,10], X[i,11], X[i,12], X[i,13], X[i,14])
            sat1.init_orbit_by_orbital_elems(*oe_observation1)
            sat2.init_orbit_by_orbital_elems(*oe_observation2)
            #OccultationCalculator
            occultation = Occultation(mars,[sat1,sat2])

            dv1, dv1_01, dv1_12, dv1_23 = earth_mars.trajectory_insertion(X[i,0], X[i,1], v_inf_vec, JS0_in, X[i,2], oe_observation1, target_is_perigee, plot_is_enabled=False)
            dv2, dv2_01, dv2_12, dv2_23= earth_mars.trajectory_insertion(X[i,0], X[i,1], v_inf_vec, JS0_in, X[i,2], oe_observation1, target_is_perigee, plot_is_enabled=False)
            f1[i] = dv1_12 + dv1_23 + dv2_12 + dv2_23

            longitude_list, latitude_list, count = occultation.simulate_position_observed(0, t0, t_end, dt)
            time, spatial = occultation.calc_evaluation(10,10,longitude_list, latitude_list)
            f2[i] = -time
            # f3[i] = -s/atial

        out["F"] = np.column_stack([f1, f2])
    
if __name__ == "__main__":
    # 問題の定義
    problem = MyProblem()
    
    # アルゴリズムの初期化（NSGA-IIを使用）
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=100,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20),
        # eliminate_duplicates=True
    )
    
    # 終了条件（40世代）
    termination = get_termination("n_gen", 40)
    
    # 最適化の実行
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    
    # 結果の可視化
    ps = problem.pareto_set(use_cache=False, flatten=False)
    pf = problem.pareto_front(use_cache=False, flatten=False)
    pop = res.pop
    
    # 目的関数空間
    plot = Scatter(title = "Objective Space")
    plot.add(pop.get("F"),color="black")
    plot.add(res.F, color = "red")
    print(res.X)
    if pf is not None:
        plot.add(pf, plot_type="line", color="red", alpha=0.7)
    plot.show()