from audioop import cross
from cmath import inf, pi
from dataclasses import replace
# from msilib.schema import Error
from platform import release
import random
import numpy as np
import warnings
from rsa import sign

class ga_options:
    def __init__(self, accuracy, bounds, fitness_function, initial_population_size, seed) -> None:
        self.accuracy = accuracy
        self.lower_bound, self.upper_bound = bounds
        self.encoding_vector = 10.0 ** (-1*np.array(range(0,accuracy+1)))
        self.fitness_function = fitness_function
        self.initial_population_size = initial_population_size
        self.rng = np.random.default_rng(seed)
        warnings.warn("Ensure fitness function is non-negative")

class chromosome():
    def __init__(self, phenotypes, ga_options) -> None:
        self.ga_options = ga_options
        assert isinstance(phenotypes, np.ndarray), "Phenotype should be a 1 dimensional Numpy.ndarray of strings"
        assert all(isinstance(phenotype, str) for phenotype in phenotypes), "All Phenotype should be string."
        for phenotype in phenotypes:
            # print(phenotype)
            assert len(phenotype) == (ga_options.accuracy+2), "Phenotype is of length {}. Expected length is {} for all phenotypes.".format(len(phenotypes), self.ga_options.accuracy+2)
        # print("*************************")
        self.phenotypes = phenotypes
        # self.phenotype_matrix = 
        
    def convert_phenotype_to_matrix(self, phenotypes):
        all_pheno_types = np.array([np.array(list(map(int,list(x)))) for x in phenotypes])
        return all_pheno_types

    def get_genotypes(self):
        phenotype_matrix = self.convert_phenotype_to_matrix(self.phenotypes)
        sign_allele = phenotype_matrix[:,0]
        # cond_1 = sign_allele!=1
        # cond_2 = sign_allele!=0
        # t = cond_1 & cond_2
        # if t.any()==True:
        #     warnings.warn('Error in sign bit. Modifying to random +1 or -1')
        #     print(sign_allele[t == True])
        sign = np.multiply(sign_allele >= 5, 1)
        sign[sign == 0] = -1
        # print(sign)
        other_alleles = phenotype_matrix[:,1:]
        real_val = np.round(sign * (other_alleles @ self.ga_options.encoding_vector.T),self.ga_options.accuracy)
        return real_val
    
    def get_fitness(self):
        genotypes = self.get_genotypes()
        fitness = np.array([self.ga_options.fitness_function(x) for x in genotypes])
        cond_1 = genotypes < self.ga_options.lower_bound
        cond_2 = genotypes > self.ga_options.upper_bound
        fitness[cond_1 | cond_2] = -1
        return fitness

    def add_phenotypes(self, new_phenotypes):
        self.phenotypes = np.hstack((self.phenotypes, new_phenotypes.phenotypes))

    def rank_by_fitness(self):
        fitness = self.get_fitness()
        return self.phenotypes[np.argsort(-1 * fitness)], fitness[np.argsort(-1 * fitness)]
    
    def get_chromosome_size(self):
        return self.phenotypes.shape[0]
    
    def reduce_population(self, number_needed):
        new_pop, _ = self.rank_by_fitness()
        return chromosome(new_pop[0:number_needed], self.ga_options)


import re
class geneticAlgorithmOperators:
    def __init__(self, ga_options):
        self.ga_options = ga_options

    def generate_legal_phenotypes(self):        
        legal_phenotypes = self.ga_options.rng.uniform(self.ga_options.lower_bound, self.ga_options.upper_bound, self.ga_options.initial_population_size)
        return chromosome(self.phenotype_to_chromosome(legal_phenotypes), self.ga_options)

    def phenotype_to_chromosome(self, phenotype):
        aes = str(abs(np.round(phenotype,self.ga_options.accuracy))).replace(".","").replace("[","").replace("]","").split()
        for iterate in range(len(phenotype)):
            if phenotype[iterate] >= 0:
                val = str(self.ga_options.rng.integers(5,10,1)[0])
            elif phenotype[iterate] < 0:
                val = str(self.ga_options.rng.integers(0,5,1)[0])
            aes[iterate] = val + aes[iterate]

            if len(aes[iterate]) < self.ga_options.accuracy+2:
                # print("0"*abs(len(aes[iterate])-self.ga_options.accuracy-2))
                aes[iterate] += "0"*abs(len(aes[iterate])-self.ga_options.accuracy-2)
        # print(np.array(aes))
        
        return np.array(aes)

    def generate_random_population(self):
        legal_phenotypes = self.generate_legal_phenotypes()
        return legal_phenotypes

    def tournament_selection(self, current_population, selection_algorithm_options):
        num_players = selection_algorithm_options["num_players"]
        number_of_offsprings = selection_algorithm_options["number_of_offsprings"]

        fitness = current_population.get_fitness()
        offsprings = []
        for _ in range(number_of_offsprings):
            select_index = self.ga_options.rng.choice(len(fitness), size = num_players, replace = False)
            offsprings.append(current_population.phenotypes[select_index[np.argmax(fitness[select_index])]])
                
        return chromosome(np.array(offsprings), self.ga_options)

    def selection(self, current_population, selection_algorithm_options):
        if selection_algorithm_options["type"] == "tournament":
            new_offsprings = self.tournament_selection(current_population, selection_algorithm_options)
        return new_offsprings, current_population

    def single_point_crossover(self, elite_children_object, number_of_offsprings):
        elite_children_phenotypes = elite_children_object.phenotypes
        elite_children_fitness = elite_children_object.get_fitness()
        crossover_offsprings = []
        for _ in range(number_of_offsprings):
            select_index = self.ga_options.rng.choice(len(elite_children_fitness), size = 2, replace = False)
            parent_1 = elite_children_phenotypes[select_index[0]]
            parent_2 = elite_children_phenotypes[select_index[1]]
            # print(parent_1)
            # print(parent_2)
            coin_toss = self.ga_options.rng.uniform(0,1,1)[0]
            
            if coin_toss < reproduce_options["crossover_probablity"]:
                # print("Crossing")
                
                point = self.ga_options.rng.choice((self.ga_options.accuracy+2), size = 1, replace=False)[0]
                # print(point)
                offspring_1 = parent_1[0:point] + parent_2[point:]
                offspring_2 = parent_2[0:point] + parent_1[point:]
                
            else:
                # print("Not Crossing")
                offspring_1 = parent_1
                offspring_2 = parent_2
            
            
            crossover_offsprings.append(offspring_1)
            crossover_offsprings.append(offspring_2)

        return chromosome(np.array(crossover_offsprings), self.ga_options)


    def mutation(self, new_population, reproduce_options):
        population_pheno = new_population.phenotypes
        
        mutated_offsprings = []
        for iterate in range(population_pheno.shape[0]):
            enc_string = population_pheno[iterate]
            temp = list(enc_string)
            coin_toss = self.ga_options.rng.uniform(0,1,len(enc_string))
            # select_index = self.ga_options.rng.choice(len(enc_string), size = len(enc_string), replace=False)
            mutation_bits = self.ga_options.rng.integers(0,10, size = len(enc_string))
            # offspring = enc_string
            # print(offspring)
            # offspring[coin_toss>reproduce_options["mutation_probablity"]] = mutation_bits[coin_toss>reproduce_options["mutation_probablity"]]
            # print(offspring)
            for iterate_2 in range(len(enc_string)):
                if coin_toss[iterate_2] < reproduce_options["mutation_probablity"]:
                    
                    temp[iterate_2] = str(mutation_bits[iterate_2])
                    
                    
            mutated_os = "".join(temp)
            # print(offspring)
            # print(coin_toss)
            # print(enc_string)
            # print(coin_toss < reproduce_options["mutation_probablity"])
            # # print(select_index)
            # print(mutation_bits)
            # print(mutated_os)
            # print(f)
            mutated_offsprings.append(mutated_os)
            

        return chromosome(np.array(mutated_offsprings), self.ga_options)

    def reproduce(self, current_population, reproduce_options):
        num_pop = current_population.get_chromosome_size()
        ranked_fit_individuals, _ = current_population.rank_by_fitness()
        elite_children_count = int(np.floor(num_pop * reproduce_options["prob_elite_children"]))
        # print(elite_children_count)
        if elite_children_count < 2:
            raise Exception("Elite children count is {}. Try increasing the prob_elite_children parameter".format(elite_children_count))
        # if num_pop - elite_children_count <= 2:
        #     warnings.warn("No population for reproduction. If this is not what you intended, try decresing prob_elite_children parameter.")
        #     elite_children = current_population
        # else:
            # print(ranked_fit_individuals)
        elite_children = chromosome(ranked_fit_individuals[0:elite_children_count], self.ga_options)
        number_of_offsprings = num_pop - elite_children_count
        crossover_offsprings = self.single_point_crossover(elite_children, number_of_offsprings)
        mutated_offsprings = self.mutation(crossover_offsprings, reproduce_options)
        elite_children.add_phenotypes(mutated_offsprings)


        return elite_children

    

def objective_function(x):
    return -1*(x * np.sin(10*np.pi*x) + 1)

# phenotypes = np.array(["12345678", "53456257", "08791232"])

def objective_function(x):
    return (x * np.sin(10*np.pi*x) + 1)

def run_ga(num_generations, ga_options, selection_algorithm_options, reproduce_options):
    

    x_genotype = []
    corres_y = []

    ga = geneticAlgorithmOperators(ga_options)
    
    pop_1 = ga.generate_legal_phenotypes()
    fitness = pop_1.get_fitness()
    
    x_genotype.append(pop_1.get_genotypes())
    corres_y.append(fitness)

    for _ in range(num_generations):
        selected_pop,_ = ga.selection(pop_1, selection_algorithm_options)
        new_pop_1 = ga.reproduce(selected_pop, reproduce_options)
        pop_1 = new_pop_1.reduce_population(options.initial_population_size)
        fitness = pop_1.get_fitness()
        x_genotype.append(pop_1.get_genotypes())
        corres_y.append(fitness)

    population = pop_1
    return population, x_genotype, corres_y



x_history = []
y_history = []
for j in range(1):
    bounds = (-0.5,1)
    accuracy = 7
    seed = 123 + j
    initial_pop_size = 5
    num_generations = 30

    options = ga_options(accuracy, bounds, objective_function, initial_pop_size, seed)

    reproduce_options = {"crossover_probablity":0.3, "prob_elite_children":0.4, "mutation_probablity":0.3}

    selection_algorithm_options = {"type":"tournament", "num_players":3, "number_of_offsprings":initial_pop_size, "replace":True}

    pop_1, x_genotype, corres_y = run_ga(num_generations, options, selection_algorithm_options, reproduce_options)
    x_history.append(x_genotype)
    y_history.append(corres_y)

x_history = np.array(x_history)
y_history = np.array(y_history)

# print(x_history.shape)
# print(y_history.shape)

import matplotlib.pyplot as plt

for iterate in range(1):
    plt.plot(np.max(y_history[iterate,:,:],1), label = "Maximum Fitness")
    plt.plot(np.min(y_history[iterate,:,:],1), label = "Minimum Fitness")
    plt.plot(np.mean(y_history[iterate,:,:],1), label = "Mean Fitness")
    plt.xlabel("Number of Generations")
    plt.ylabel("Fitness")
    
plt.legend()
plt.tight_layout()
plt.savefig("min_max_mean_q1", dpi = 500, format = 'png')
# plt.show()

from matplotlib import animation
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(xlim=(-0.5,1.), ylim=(-0.25, 2))
x = np.linspace(-0.5,1,10000)
y = objective_function(x)
line_1, = ax.plot(x,y, label = "Objective Function")
ax.plot(x_history[0,-1,:], y_history[0,-1,:],".k", markersize = 5, label = "Points")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()
fig.tight_layout()
plt.savefig("Evalute_best_point_q1", dpi = 500, format = 'png')
# plt.show()


fig = plt.figure()
ax = plt.axes()

x_best = []
for iter, y in enumerate(y_history[0,:,:]):
    ind = np.argmax(y)
    x_best.append(x_history[0,iter,ind])

ax.plot(x_best, label = "Trajectory of best point in every generation")
ax.plot(x_best, "ok", markersize = 5, label = "Points")
ax.set_xlabel("Number of Generation")
ax.set_ylabel("Best Point")
ax.legend()
fig.tight_layout()
plt.savefig("point_trajectory_q1", dpi = 500, format = 'png')




# line, = ax.plot([], [], ".k", markersize = 10)
# time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)



# def animate(i):
#     print(i)
#     x = x_history[0,i,:]
#     y = y_history[0,i,:]
#     line.set_data(x, y)
#     time_text.set_text("Generation {}".format(i))
#     return line, [time_text,],


# anim = animation.FuncAnimation(fig, animate, frames=num_generations, interval=201, blit=False)
# # ax.legend()
# plt.show()

print(x_history[0,-1,:])
print(y_history[0,-1,:])