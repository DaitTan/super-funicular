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
        assert all(isinstance(phenotype, float) for phenotype in phenotypes), "All Phenotype should be string."
        self.phenotypes = phenotypes
        # self.phenotype_matrix = 
        
    def convert_phenotype_to_matrix(self, phenotypes):
        all_pheno_types = np.array([np.array(list(map(int,list(x)))) for x in phenotypes])
        return all_pheno_types

    def get_genotypes(self):
        real_val = self.phenotypes
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
        return chromosome(legal_phenotypes, self.ga_options)

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

    def single_point_crossover(self, elite_children_object, number_of_offsprings, reproduce_options):
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
            
            if coin_toss > reproduce_options["crossover_probablity"]:
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
        noise_mean = reproduce_options["mutation_noise_mean"]
        noise_std = reproduce_options["mutation_noise_std"]
        mut = self.ga_options.rng.normal(noise_mean, noise_std, size = population_pheno.shape[0])
        mutated_offsprings = population_pheno + mut
        # print(population_pheno)
        # print(mut)
        # print(mutated_offsprings)

        # print(f)
            

        return chromosome(np.array(mutated_offsprings), self.ga_options)

    def reproduce(self, current_population, reproduce_options):
        num_pop = current_population.get_chromosome_size()
        ranked_fit_individuals, _ = current_population.rank_by_fitness()
        elite_children_count = int(np.floor(num_pop * reproduce_options["prob_elite_children"]))
        # print(elite_children_count)
        if elite_children_count < 2:
            raise Exception("Elite children count is {}. Try increasing the prob_elite_children parameter".format(elite_children_count))
            # print(ranked_fit_individuals)
        elite_children = chromosome(ranked_fit_individuals[0:elite_children_count], self.ga_options)
        number_of_offsprings = num_pop - elite_children_count
        crossover_offsprings = self.single_point_crossover(elite_children, number_of_offsprings, reproduce_options)
        mutated_offsprings = self.mutation(crossover_offsprings, reproduce_options)
        elite_children.add_phenotypes(mutated_offsprings)


        return elite_children

    

def objective_function(x):
    return (x * np.sin(10*np.pi*x) + 1)

phenotypes = np.array([1.23, 3.456, 3.1232, -.2345])
bounds = (-0.5,1)
accuracy = 10
seed = 123
initial_pop_size = 5
options = ga_options(accuracy, bounds, objective_function, initial_pop_size, seed)

reproduce_options = {"crossover_probablity":1, "prob_elite_children":0.4, "mutation_noise_mean":0, "mutation_noise_std":0.01}

selection_algorithm_options = {"type":"tournament", "num_players":3, "number_of_offsprings":5, "replace":True}


hist = []
hist_mean = []
hist_max = []
hist_min = []


ga = geneticAlgorithmOperators(options)
num_generations = 1000
pop_1 = ga.generate_legal_phenotypes()
fitness = pop_1.get_fitness()
max_fit = np.argmax(fitness)
min_fit = np.argmin(fitness)
mean_fit = np.mean(fitness)
hist_max.append(max_fit)
hist_min.append(min_fit)
hist_mean.append(mean_fit)

x_genotype = []
corres_y = []
x_genotype.append(pop_1.get_genotypes())
corres_y.append(pop_1.get_fitness())

for iterate in range(num_generations):
    new_pop_1 = ga.reproduce(pop_1, reproduce_options)
    pop_1 = new_pop_1.reduce_population(options.initial_population_size)
    fitness = pop_1.get_fitness()
    max_fit = np.argmax(fitness)
    min_fit = np.argmin(fitness)
    mean_fit = np.mean(fitness)
    hist_max.append(max_fit)
    hist_min.append(min_fit)
    hist_mean.append(mean_fit)
    x_genotype.append(pop_1.get_genotypes())
    corres_y.append(pop_1.get_fitness())




from matplotlib import animation
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(xlim=(-0.5,1.), ylim=(-0.25, 2))
x = np.linspace(-0.5,1,10000)
y = objective_function(x)
line_1, = ax.plot(x,y)
line, = ax.plot([], [], ".k", markersize = 10)
time_text = ax.text(0.05, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)



def animate(i):
    x = x_genotype[i]
    y = corres_y[i]
    line.set_data(x, y)
    time_text.set_text("Generation {}".format(i))
    return line, [time_text,],


anim = animation.FuncAnimation(fig, animate, frames=num_generations, interval=200, blit=False)
# ax.legend()
plt.show()

# plt.show()
plt.plot(hist_max, label = "max fitness")
plt.plot(hist_min, label = "min fitness")
plt.plot(hist_mean, label = "mean fitness")
plt.legend()
plt.show()

phenotype, fitness = pop_1.rank_by_fitness()

print(chromosome(phenotype,options).get_genotypes())
print(fitness)


