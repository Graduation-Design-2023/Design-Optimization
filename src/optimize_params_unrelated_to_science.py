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
    JS0_in = 0 # fix me

    deg_max = 180
    r_h_max = 10 * mars.radius
    v_in_max = 100 * (mars.mu / mars.radius**3)**0.5 # ?
    r_a_max = 100 * mars.radius

    def obfunc(individual):
        theta = individual[0] * deg_max
        r_h = individual[1] * r_h_max
        v_in_x = individual[2] * v_in_max
        v_in_y = individual[3] * v_in_max
        v_in_z = individual[4] * v_in_max
        v_in_vec = np.array([[v_in_x], [v_in_y], [v_in_z]])
        r_a = individual[5] * r_a_max
        
        if(r_a < r_h or r_h < mars.radius or not(-deg_max < theta < deg_max)):
            object = np.inf
        else:
            object = earth_mars.trajectory_insertion(theta, r_h, v_in_vec, JS0_in, r_a, oe_observation, plot_is_enabled=False)

        return object,
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_gene", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_gene, 6)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate", tools.cxBlend,alpha=0.2)
    toolbox.register("mutate", tools.mutGaussian, mu=[0.0, 0.0], sigma=[20.0, 20.0], indpb=0.2)
    toolbox.register("evaluate", obfunc)
    #以下でパラメータの設定
    #今回は最も単純な遺伝的アルゴリズムの手法を採用
    #乱数を固定
    random.seed(64)
    #何世代まで行うか
    NGEN = 50
    #集団の個体数
    POP = 80
    #交叉確率
    CXPB = 0.9
    #個体が突然変異を起こす確率
    MUTPB = 0.1
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