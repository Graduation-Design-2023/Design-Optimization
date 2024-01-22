
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

# values for params' domain
windows = (2040, 1, 1, 0, 0, 0)
search_period = 2 * 365 * 24 * 60 * 60
trans_period_max = 5 * 365 * 24 * 60 * 60
print(windows)
JS_launch_min = myval.convert_times_to_T_TDB(*windows)[0] - search_period
JS_launch_max = myval.convert_times_to_T_TDB(*windows)[0] + search_period
v_max = (myval.mu("Sun") / myval.a("Mars", JS_launch_min))**0.5 * 1.5

xl_array = np.array([JS_launch_min, 
                     JS_launch_min, 
                     JS_launch_min, 
                     -v_max, 
                     -v_max, 
                     -v_max,
                     -v_max, 
                     -v_max, 
                     -v_max,
                     0,
                     0,
                     JS_launch_min,
                     JS_launch_min])
xu_array = np.array([JS_launch_max, 
                     JS_launch_max+trans_period_max,
                     JS_launch_max+trans_period_max, 
                     v_max, 
                     v_max, 
                     v_max,
                     v_max, 
                     v_max, 
                     v_max,
                     360,
                     2 * mars.radius,
                     JS_launch_max+trans_period_max,
                     JS_launch_max+trans_period_max])

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=13,
                         n_obj=1,
                         n_constr=4,
                         xl=xl_array,
                         xu=xu_array,
                        )
        
    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X[:,0])
        f1 = np.zeros(n)
        g1 = np.zeros(n)
        g2  = np.zeros(n)
        g3  = np.zeros(n)
        g4  = np.zeros(n)

        for i in range(0, n):
            JS_end = X[i,0]
            JS_swingby = X[i,1]
            JS_launch = X[i,2]
            arrival_JS_list = [JS_end, JS_swingby, JS_launch]
            v_inf_end = np.array([[X[i,3]], [X[i,4]], [X[i,5]]])
            v_inf_swingby = np.array([[X[i,6]], [X[i,7]], [X[i,8]]])
            v_inf_in_list = [v_inf_end, v_inf_swingby]
            theta = X[i,9]
            h = X[i,10]
            JS_tcm01 = X[i,11]
            JS_tcm12 = X[i,12]
            JS_tcm_list = [JS_tcm01, JS_tcm12]

            _, _, _, _, f1[i] = earth_mars.trajectory_with_swingby([mars],arrival_JS_list, v_inf_in_list, [theta], [h], JS_tcm_list)

            g1[i] = JS_tcm01 - JS_end
            g2[i] = JS_swingby - JS_tcm01
            g3[i] = JS_tcm12 - JS_swingby
            g4[i] = JS_launch - JS_tcm12

        out["F"] = np.column_stack([f1])
        out["G"] = np.column_stack([g1, g2, g3, g4])

if __name__ == '__main__':
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
    termination = get_termination("n_gen", 60)
    
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

    i = 0
    X = np.array([res.X])
    JS_end = X[i,0]
    JS_swingby = X[i,1]
    JS_launch = X[i,2]
    arrival_JS_list = [JS_end, JS_swingby, JS_launch]
    v_inf_end = np.array([[X[i,3]], [X[i,4]], [X[i,5]]])
    v_inf_swingby = np.array([[X[i,6]], [X[i,7]], [X[i,8]]])
    v_inf_in_list = [v_inf_end, v_inf_swingby]
    theta = X[i,9]
    h = X[i,10]
    JS_tcm01 = X[i,11]
    JS_tcm12 = X[i,12]
    JS_tcm_list = [JS_tcm01, JS_tcm12]

    earth_mars.trajectory_with_swingby([mars],arrival_JS_list, v_inf_in_list, [theta], [h], JS_tcm_list, plot_is_enabled = True)