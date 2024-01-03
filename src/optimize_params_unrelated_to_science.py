from libs.lib import Values, TrajectoryCalculator, Planet
from libs.occultation_lib import Satellite, Occultation
from libs.interplanetary_lib import PlanetsTransOrbit

import numpy as np
from matplotlib import pyplot as plt
import random
from deap import base, creator, tools, algorithms
from deap.benchmarks import ackley

if __name__ == '__main__':
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

    a = mars.radius + 2124
    e = 0.2231
    i = 80
    omega = 50
    Omega = 0
    tp= 0 #invalid
    oe_observation = (a, e, i, omega, Omega, tp)

    windows , t_H = earth_mars.calc_launch_window(2024, 4, 1, 0.001, 1)
    JS_launch, _, _ = myval.convert_times_to_T_TDB(2024, 4, 1, 0, 0, 0)
    duration = t_H
    JS0_in = JS_launch + duration

    _,_,_,_,v_planet_start,v_planet_end,v_sat_start,v_sat_end = earth_mars.trajectory_by_lambert(windows[0], duration)
    v_inf_vec = v_sat_end - v_planet_end

    deg_max = 360
    r_h_max = 10 * mars.radius
    v_in_max = 100 * (mars.mu / mars.radius**3)**0.5 # ?
    r_a_max = 100 * mars.radius
    target_is_perigee = False

    def obfunc(individual):
        theta = individual[0] * deg_max
        r_h = individual[1] * r_h_max
        r_a = individual[2] * r_a_max
        
        if(r_a < r_h or r_h < mars.radius or not(0 < theta < deg_max)):
            object = np.inf
        else:
            delta_v_tot,delta_v01 ,delta_v12 ,delta_v23 = earth_mars.trajectory_insertion(theta, r_h, v_inf_vec, JS0_in, r_a, oe_observation,target_is_perigee, plot_is_enabled=False)
            object = delta_v_tot

        return object,
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_gene", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_gene, 3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", tools.cxBlend,alpha=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=[0.5, 0.5, 0.5], sigma=[1.,1.,1.], indpb=0.2)
    toolbox.register("evaluate", obfunc)
    #以下でパラメータの設定
    #今回は最も単純な遺伝的アルゴリズムの手法を採用
    #乱数を固定
    random.seed(64)
    #何世代まで行うか
    NGEN = 20
    #集団の個体数
    POP = 1000
    #交叉確率
    CXPB = 0.9
    #個体が突然変異を起こす確率
    MUTPB = 0.3
    #集団は80個体という情報の設定
    pop = toolbox.population(n=POP)
    #集団内の個体それぞれの適応度（目的関数の値）を計算
    for individual in pop:
        individual.fitness.values = toolbox.evaluate(individual)
    #パレート曲線上の個体(つまり、良い結果の個体)をhofという変数に格納
    hof = tools.ParetoFront()
    #今回は最も単純なSimple GAという進化戦略を採用
    algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, halloffame=hof)
    #最終的な集団(pop)からベストな個体を1体選出する関数
    best_ind = tools.selBest(pop, 1)[0]
    #結果表示
    print("最も良い個体は %sで、そのときの目的関数の値は %s" % (best_ind, best_ind.fitness.values))

    theta = best_ind[0] * deg_max
    r_h = best_ind[1] * r_h_max
    r_a = best_ind[2] * r_a_max
    val = earth_mars.trajectory_insertion(theta, r_h, v_inf_vec, JS0_in, r_a, oe_observation,target_is_perigee)
    print(val)