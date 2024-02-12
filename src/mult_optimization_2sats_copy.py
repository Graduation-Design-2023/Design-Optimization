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
sat1 = Satellite(mars, receiver_is_avalable=True)
sat2 = Satellite(mars, receiver_is_avalable=True)
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
v_inf_vec = np.array([[1.0432425892660173,1.2835465382459306,0.24612132122768177]]).T

result_file_name = "outputs/runs/obs.json"
with open(result_file_name, "r") as f:
    results = json.load(f)
    result = results[0]

print(result)
oe_observation1 = tuple(result[1:7])
oe_observation2 = tuple(result[7:])
#OccultationCalculator
occultation = Occultation(mars,[sat1,sat2])

# ----------------------------
# X : [theta, r_h, r_a, Omega]
xl_array = np.array([0, radius, radius, 0])
xu_array = np.array([360, radius*10, radius*10, 360])

class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=4,
                         n_obj=1,
                         n_constr=0,
                         xl=xl_array,
                         xu=xu_array,
                        )
        
    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X[:,0])
        f1 = np.zeros(n)
        for i in range(0, n):
            result[5] = X[i,3]
            result[11] = X[i,3]
            oe_observation1 = tuple(result[1:7])
            oe_observation2 = tuple(result[7:])
            dv = earth_mars.trajectory_insertion_2sats(X[i,0], X[i,1], v_inf_vec, JS0_in, X[i,2], oe_observation1,oe_observation2, target_is_perigee,plot_is_enabled=False)
            f1[i] = dv

        out["F"] = np.column_stack([f1])
    
if __name__ == "__main__":
    now = datetime.now()
    folder_name = "outputs/runs"
    file_name = f"{folder_name}/{now.strftime('%Y%m%d%H%M%S')}.json"
    os.makedirs(folder_name, exist_ok=True)

    # 問題の定義
    problem = MyProblem()
    
    # アルゴリズムの初期化（NSGA-IIを使用）
    algorithm = NSGA2(
        pop_size=50,
        n_offsprings=50,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(eta=20),
        # eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", 50)
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    
    ps = problem.pareto_set(use_cache=False, flatten=False)
    pf = problem.pareto_front(use_cache=False, flatten=False)
    pop = res.pop
    X = res.X
    print('X: ', X)
    result[5] = X[3]
    result[11] = X[3]
    oe_observation1 = tuple(result[1:7])
    oe_observation2 = tuple(result[7:])
    # dv, _, _, _ = earth_mars.trajectory_insertion(X[0], X[1], v_inf_vec, JS0_in, X[2], oe_observation1, target_is_perigee)
    dv = earth_mars.trajectory_insertion_2sats(X[0], X[1], v_inf_vec, JS0_in, X[2], oe_observation1,oe_observation2, target_is_perigee)
