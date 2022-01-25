from cmath import inf, pi
from dataclasses import replace
from platform import release
import random
from git import typ
import numpy as np
from rsa import sign

class ga_options:
    def __init__(self, accuracy, bounds, fitness_function) -> None:
        self.accuracy = accuracy
        self.lower_bound, self.upper_bound = bounds
        self.encoding_vector = 10.0 ** (-1*np.array(range(0,accuracy+1)))
        self.fitness_function = fitness_function
        Warning("Ensure fitness function is non-negative")

class chromosome(object):
    def __init__(self, phenotypes, ga_options) -> None:
        self.ga_options = ga_options
        assert isinstance(phenotypes, np.ndarray), "Phenotype should be a 1 dimensional Numpy.ndarray of strings"
        assert all(isinstance(phenotype, str) for phenotype in phenotypes), "All Phenotype should be string."
        assert all(len(phenotype) == ga_options.accuracy+2 for phenotype in phenotypes), "Phenotype is of length {}. Expected length is {} for all phenotypes.".format(len(phenotypes), self.ga_options.accuracy+2)
        
        self.phenotype_matrix = self.convert_phenotype_to_matrix(phenotypes)
    
    def convert_phenotype_to_matrix(self, phenotypes):
        all_pheno_types = np.array([np.array(list(map(int,list(x)))) for x in phenotypes])
        return all_pheno_types

    def get_genotypes(self):
        sign_allele = self.phenotype_matrix[:,0]
        sign = np.multiply(sign_allele >= 5, 1)
        sign[sign == 0] = -1
        other_alleles = self.phenotype_matrix[:,1:]
        real_val = np.round(sign * (other_alleles @ self.ga_options.encoding_vector.T),self.ga_options.accuracy)
        return real_val
    
    def get_fitness(self):
        genotypes = self.get_genotypes()
        fitness = np.array([self.ga_options.fitness_function(x) for x in genotypes])
        print(fitness)
        cond_1 = fitness < self.ga_options.lower_bound
        cond_2 = fitness > self.ga_options.upper_bound
        # print()
        fitness[cond_1 | cond_2] = -1
        print(fitness)
        return fitness

def objective_function(x):
    return -1*(x * np.sin(10*np.pi*x) + 1)

phenotypes = np.array(["12345678", "43456257", "58791232"])
bounds = (-1,1)
accuracy = 6
options = ga_options(accuracy, bounds, objective_function)

x = chromosome(phenotypes, options)
x.get_fitness()

# class population_utils(chromosome):
#     def __init__(self, population_size, objective_function, accuracy, bounds, random_seed = 42):
#         self.accuracy = accuracy
#         self.lower_bound, self.upper_bound = bounds
#         self.population_size = population_size
#         self.rng = np.random.default_rng(random_seed)
#         self.generate_legal_phenotypes()
#         self.initial_population = self.initialize_population()
#         self.current_population = self.initial_population.copy()
#         self.objective_function = objective_function
    
#     def set_population(self, new_population):
#         # print(new_population)
#         self.current_population = new_population
    
#     def get_population(self):
#         return self.current_population

#     def thanos_kill_population(self):
#         fitness = self.get_population_fitness(self.current_population)
#         sorted_fitness = np.argsort(fitness)
#         survived_population = [self.current_population[iterate] for iterate in sorted_fitness[0:len(self.initial_population)]]
#         self.set_population(survived_population)
        
#     def generate_legal_phenotypes(self):
#         legal_phenotypes = self.rng.uniform(self.lower_bound, self.upper_bound, self.population_size)
#         return legal_phenotypes

#     def phenotype_to_chromosome(self, phenotype):
#         if phenotype > self.upper_bound or phenotype < self.lower_bound:
#             raise Exception("Illegal phentoype for conversion")
#         enc_string = ""
#         if phenotype >= 0:
#             enc_string += str(1)
#         elif phenotype < 0:
#             enc_string += str(0)

#         enc_string += str(abs(np.round(phenotype,self.accuracy))).replace(".","")
#         # print(len(enc_string))
#         if len(enc_string) < self.accuracy+2:
#             # print("HAHA")
#             # v = "H" * abs(len(enc_string)-self.accuracy-2)
#             # print(v)
#             enc_string += "0"*abs(len(enc_string)-self.accuracy-2)

#         return enc_string

#     def initialize_population(self):
#         legal_phenotypes = self.generate_legal_phenotypes()
#         # print(legal_phenotypes)
#         # print([self.phenotype_to_chromosome(iterate)for iterate in legal_phenotypes])
#         initial_population = [chromosome(self.phenotype_to_chromosome(iterate) ,self.accuracy, (self.lower_bound, self.upper_bound)) for iterate in legal_phenotypes]
#         return initial_population

#     def get_population_chromosomes(self,current_population):
#         return [iterate.enc_string for iterate in current_population]
    
#     def get_population_fitness(self,current_population):
#         return [self.objective_function(np.round(iterate.phenotype, self.accuracy)) for iterate in current_population]
    
#     def get_population_phenotypes(self, current_population):
#         return [np.round(iterate.phenotype, self.accuracy) for iterate in current_population]



#     def selection(self, current_population, selection_algorithm_options):
#         temp_init_population = current_population.copy()
        
#         if selection_algorithm_options["type"] == "tournament":
#             num_players = selection_algorithm_options["num_players"]
#             number_of_offsprings = selection_algorithm_options["number_of_offsprings"]

#             if selection_algorithm_options["replace"] == False:
#                 assert (len(temp_init_population) >= number_of_offsprings + num_players-1), \
#                     "Replace option is set to {}. Expected: Length(population)(={}) >= number_of_offsprings (={}) + number_of_players (={}) - 1\n number_of_offspring <= {}".format(selection_algorithm_options["replace"], len(temp_init_population), number_of_offsprings, num_players, len(temp_init_population)-num_players+1)
            
#             self.rng.shuffle(temp_init_population)
#             # print(current_population)
#             # print(self.get_population_fitness(current_population))

#             offsprings = []
#             for _ in range(number_of_offsprings):
#                 select_index = self.rng.choice(len(temp_init_population), size = num_players, replace = False)
#                 # print(select_index)
#                 fitness = self.get_population_fitness([temp_init_population[iterate] for iterate in select_index])
#                 # print(fitness)
#                 offsprings.append(temp_init_population[select_index[np.argmin(fitness)]])
#                 if selection_algorithm_options["replace"] == False:
#                     del temp_init_population[select_index[np.argmin(fitness)]]

#         return offsprings

#     def crossover_operator(self, current_population, crossover_operator_options, selection_algorithm_options):
#         temp_curr_population = current_population.copy()
#         if crossover_operator_options["type"] == "single_point_crossover":
#             self.rng.shuffle(temp_curr_population)
#             selected_chromosome = self.selection(temp_curr_population, selection_algorithm_options)
#             # print(selected_chromosome)
#             chromosome_string = self.get_population_chromosomes(selected_chromosome)
#             # print("*************")
#             # print(self.accuracy+2)
#             point = self.rng.choice((self.accuracy+2), size = 1, replace=False)[0]
#             # print(point)
#             temp_chromosome = chromosome_string[0]
#             chromosome_1 = chromosome(chromosome_string[0][0:point] + chromosome_string[1][point:], self.accuracy, (self.lower_bound, self.upper_bound))
#             chromosome_2 = chromosome(chromosome_string[1][0:point] + temp_chromosome[point:], self.accuracy, (self.lower_bound, self.upper_bound))
#             return [chromosome_1, chromosome_2]

#     def mutation_operator(self, current_population, mutation_algorithm_options):
#         temp_curr_population = current_population.copy()
#         if mutation_algorithm_options["type"] == "swap_mutation":
#             prob_mutation = mutation_algorithm_options["prob_mutation"]
#             mutated_chromosomes = []
#             for iterate_chromosome in temp_curr_population:
#                 enc_string = iterate_chromosome.enc_string
#                 rand_int = self.rng.uniform(size = 1)[0]
#                 if rand_int < prob_mutation:
#                     swap_indices = self.rng.choice(len(enc_string), size = 2, replace = False)
#                     # print(enc_string)
#                     indices = np.arange(len(enc_string))
#                     # print("*******************************")
#                     # print(indices)
#                     # print(swap_indices)
#                     lower = np.min(swap_indices)
#                     upper = np.max(swap_indices)+1
#                     subset = np.array(indices[lower:upper])
#                     shuffled_subset = self.rng.permutation(subset)
#                     # print(shuffled_subset)
#                     indices[lower:upper] = shuffled_subset
#                     mutated_chromosome = "".join([enc_string[iterate] for iterate in indices])
#                     mutated_chromosomes.append(chromosome(mutated_chromosome, self.accuracy, (self.lower_bound, self.upper_bound)))
#                     # print(indices)
#                     # print(enc_string)
#                     # print(mutated_chromosome)
#                     # print("***********************************")
#                     # print(f)
#                     # self.rng.shuffle(enc_string[shu])
#                     # print(enc_string)
#                 else:
#                     mutated_chromosomes.append(iterate_chromosome)
        
#         return mutated_chromosomes
            
                
# def objective_function(x):
#     return -1*(x * np.sin(10*np.pi*x) + 1)

# # print(f)
# accuracy = 5
# bounds = (-5,5)
# a = population_utils(1000,objective_function, accuracy, bounds, 45)
# # print(a.get_population_fitness(a.initial_population))
# # print(a.get_population_chromosomes(a.initial_population))


# selection_algorithm_options = {"type":"tournament", "num_players":3, "number_of_offsprings":100, "replace":True}
# crossover_algorithm_options = {"type": "single_point_crossover"}
# mutation_algorithm_options = {"type": "swap_mutation", "prob_mutation":0.9}

# # selct = a.selection(a.initial_population, selection_algorithm_options)
# # print(a.get_population_fitness(selct))
# hist = []
# for i in range(100):
#     print(i)
#     popula = a.get_population()

#     crossed = a.crossover_operator(popula, crossover_algorithm_options, selection_algorithm_options)
#     a.set_population(popula+crossed)

#     popula = a.get_population()
#     mutated = a.mutation_operator(popula, mutation_algorithm_options)
#     a.set_population(mutated)
#     a.thanos_kill_population()
#     new_popu = a.get_population()

#     # print(a.get_population_fitness(a.initial_population))
#     hist.append(np.sort(a.get_population_fitness(new_popu))[0])
#     print("*******************************")

# import matplotlib.pyplot as plt

# plt.plot(hist)
# plt.show()
# print(a.get_population_fitness(new_popu))


