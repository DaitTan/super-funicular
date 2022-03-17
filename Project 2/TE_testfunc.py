
import numpy as np
import pathlib

from driver import Objectives, Population, GARoutine, NonDominatedSorting
import pickle
from tqdm import tqdm
from numpy.random import default_rng

# def obj_1(pop):
#     x = pop[:, 0]
#     return x

# def obj_2(pop):
#     y = pop[:,1:]
#     # print(np.sum(y,1))
#     # print(f)
#     g = 1 + (9/29) * (np.sum(y,1))
#     h = 1 - (pop[:,0]/g)**2

#     return 1*g*h

def obj_1(pop):
    x = pop[:, 0]

    return -1 * (x * (np.sin(10 * np.pi * x)))

def obj_2(pop):
    x = pop[:, 0]
    return -1 * (2.5 * x * (np.cos(3 * np.pi * x)))



objective_1 = {}
objective_1["type"] = "Minimize"
objective_1["function"] = obj_1

objective_2 = {}
objective_2["type"] = "Minimize"
objective_2["function"] = obj_2


objectives_list = [objective_1, objective_2]
obj_name = "test_ex"




objectives = Objectives(objectives_list)


population_size = 100
num_variables = 1
bounds = [[0,1]]*1

crossover_prob = 0.9
mutation_prob = 0.15

p_curve_param = 20
p_curve_param_mutation = 20

num_generations = 100
num_runs = 1




base_path = pathlib.Path()
result_directory = base_path.joinpath(obj_name)
result_directory.mkdir(exist_ok=True)

for run in tqdm(range(num_runs)):
    seed = 123458 + run
    rng = default_rng(seed)
    run_folder = result_directory.joinpath(obj_name + "_Run_"+str(run) + "_seed_"+str(seed))
    run_folder.mkdir(exist_ok=True)

    pop = Population(population_size, num_variables, bounds, objectives, rng.integers(1e16))

    file_name = obj_name + "_Run_" + str(run) +"_initial_pop.pkl"
    f = open(run_folder.joinpath(file_name), "wb")
    pickle.dump(pop,f)
    f.close()

    
    for generation in tqdm(range(num_generations)):
        
        
        fds = NonDominatedSorting(pop)
        fds.crowding_distance(pop)
        
        
        ga = GARoutine(pop, rng.integers(1e16))
        sel_pop = ga.crowded_binary_tournament_selection()
        crossover_offsprings = ga.sbx_crossover_operator(sel_pop, crossover_prob, p_curve_param)
        mut_offspring = ga.polynomial_mutation_operator(crossover_offsprings, bounds, mutation_prob, p_curve_param_mutation)

        new_sol_vecs = np.vstack((np.array(pop.get_all_sol_vecs()), mut_offspring))


        temp_extended_pop = Population(population_size, num_variables, bounds, objectives, rng.integers(1e16), new_sol_vecs, False)
        fds = NonDominatedSorting(temp_extended_pop)
        fds.crowding_distance(temp_extended_pop)
    
        new_pop = temp_extended_pop.thanos_kill_move()
        pop = new_pop

        file_name = obj_name + "_Gen_" + str(generation) + "_Run_" + str(run) +".pkl"
        f = open(run_folder.joinpath(file_name), "wb")
        pickle.dump(pop.get_all_sol_vecs(),f)
        f.close()

