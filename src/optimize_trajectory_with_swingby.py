
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

import json
import os 
from datetime import datetime

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

# change below values--------------------------------------------------------------
# planet used for swingby
planets_swingby = [mars, mars, mars, earth, earth] # インデックスが小さいほど時間的にあと
# values for params' domain
windows = (2040, 1, 1, 0, 0, 0)
search_period = 5 * 365 * 24 * 60 * 60
trans_period_max = 5 * 365 * 24 * 60 * 60
print(windows)
JS_launch_min = myval.convert_times_to_T_TDB(*windows)[0] - search_period
JS_launch_max = myval.convert_times_to_T_TDB(*windows)[0] + search_period
v_max = (myval.mu("Sun") / myval.a("Mars", JS_launch_min))**0.5 * 1.5
#----------------------------------------------------------------------------------

num_swingby = len(planets_swingby) - 2
"""
Xのパーティション分配
    arrival_JS_list = [] # num_swingby+2
    v_inf_in_list = [] # 3*(num_swingby+1)
    theta_list = [] # num_swingby
    h_list = [] # num_swingby
    JS_tcm_list = [] # num_swingby+1
"""

def append_scholar_to_list(list, num_swingby, val):
    for j in range(num_swingby):
        list.append(val)

# make xl list
xl_list = []
append_scholar_to_list(xl_list, num_swingby+2, JS_launch_min)
append_scholar_to_list(xl_list, (num_swingby+1)*3, -v_max)
append_scholar_to_list(xl_list, num_swingby, 0)
append_scholar_to_list(xl_list, num_swingby, 0)
append_scholar_to_list(xl_list, num_swingby+1, JS_launch_min)
xl_array = np.array(xl_list)

# make xu list
xu_list = []
append_scholar_to_list(xu_list, 1, JS_launch_max)
append_scholar_to_list(xu_list, num_swingby+1, JS_launch_max+trans_period_max)
append_scholar_to_list(xu_list, (num_swingby+1)*3, v_max)
append_scholar_to_list(xu_list, num_swingby, 360)
append_scholar_to_list(xu_list, num_swingby, 2 * mars.radius)
append_scholar_to_list(xu_list, num_swingby+1, JS_launch_max+trans_period_max)
xu_array = np.array(xu_list)


# functions for converting X to arguments of trajectory_with_swingby
def append_X_scholar_to_list(list, num_swingby, X_i, j_start):
    for j in range(num_swingby):
        list.append(X_i[j_start+j])
    return j+j_start

def append_X_ndarray3_to_list(list, num_swingby, X_i, j_start):
    for j in range(num_swingby):
        vec = np.array([[X_i[j_start+3*j]], [X_i[j_start+3*j+1]], [X_i[j_start+3*j+2]]])
        list.append(vec)
    return j_start+3*j+2


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=7*num_swingby+6,
                         n_obj=2,
                         n_constr=2*(num_swingby+1),
                         xl=xl_array,
                         xu=xu_array,
                        )
        
    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X[:,0])
        f1 = np.zeros(n)
        f2 = np.zeros(n)
        less_than_0 = np.zeros((n,2*(num_swingby+1)))

        for i in range(0, n):
            j = 0
            arrival_JS_list = [] # num_swingby+2
            v_inf_in_list = [] # 3*(num_swingby+1)
            theta_list = [] # num_swingby
            h_list = [] # num_swingby
            JS_tcm_list = [] # num_swingby+1

            j = append_X_scholar_to_list(arrival_JS_list, num_swingby+2, X[i], j)
            j = append_X_ndarray3_to_list(v_inf_in_list, num_swingby+1, X[i], j+1)
            j = append_X_scholar_to_list(theta_list, num_swingby, X[i], j+1)
            j = append_X_scholar_to_list(h_list, num_swingby, X[i], j+1)
            j = append_X_scholar_to_list(JS_tcm_list, num_swingby+1, X[i], j+1)

            _, _, f2[i], _, f1[i] = earth_mars.trajectory_with_swingby(planets_swingby,arrival_JS_list, v_inf_in_list, theta_list, h_list, JS_tcm_list)
            f1[i] -= f2[i]

            # 不等式制約を作るために時系列（インデックスが小さいほど後）に並べる
            JS_tot_list = [0] * (num_swingby*2 + 3)
            JS_tot_list[::2] = arrival_JS_list
            JS_tot_list[1::2] = JS_tcm_list
            # [JS_tcm01 - JS_end, JS_swingby0 - JS_tcm01, JS_tcm12 - JS_swingby0, ...]
            less_than_0[i,:] = np.array(JS_tot_list[1:]) - np.array(JS_tot_list[:-1])

        out["F"] = np.column_stack([f1, f2])
        out["G"] = less_than_0

if __name__ == '__main__':
    now = datetime.now()
    folder_name = "outputs/runs"
    file_name = f"{folder_name}/{now.strftime('%Y%m%d%H%M%S')}.json"
    os.makedirs(folder_name, exist_ok=True)

    # 問題の定義
    problem = MyProblem()
    
    # アルゴリズムの初期化（NSGA-IIを使用）
    algorithm = NSGA2(
        pop_size=1000,
        n_offsprings=1000,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20),
        # eliminate_duplicates=True
    )
    
    # 終了条件（40世代）
    termination = get_termination("n_gen", 300)
    
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
    # plot.add(pop.get("F"),color="black")
    plot.add(res.F, color = "red")
    print(res.X)
    if pf is not None:
        plot.add(pf, plot_type="line", color="red", alpha=0.7)
    plot.show()

    output = np.concatenate([res.F, res.X], axis=1)
    print(output)
    with open(file_name, "w") as f:
        json.dump(output.tolist(), f, indent=" ") 

    i = 0
    X = res.X
    j = 0
    arrival_JS_list = [] # num_swingby+2
    v_inf_in_list = [] # 3*(num_swingby+1)
    theta_list = [] # num_swingby
    h_list = [] # num_swingby
    JS_tcm_list = [] # num_swingby+1

    j = append_X_scholar_to_list(arrival_JS_list, num_swingby+2, X[i], j)
    j = append_X_ndarray3_to_list(v_inf_in_list, num_swingby+1, X[i], j+1)
    j = append_X_scholar_to_list(theta_list, num_swingby, X[i], j+1)
    j = append_X_scholar_to_list(h_list, num_swingby, X[i], j+1)
    j = append_X_scholar_to_list(JS_tcm_list, num_swingby+1, X[i], j+1)

    res = earth_mars.trajectory_with_swingby(planets_swingby,arrival_JS_list, v_inf_in_list, theta_list, h_list, JS_tcm_list, plot_is_enabled="True")
    print(res)
