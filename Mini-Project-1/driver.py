from cmath import inf
import random
import numpy as np


class chromosome(object):
    def __init__(self, enc_string, accuracy, bounds) -> None:
        self.accuracy = accuracy
        self.lower_bound, self.upper_bound = bounds
        assert len(enc_string) == self.accuracy+2, "Encoding String is of length {}. Expected length is {}.".format(len(enc_string), accuracy+2)
        # self.encoding_vector = 10.0 ** (-1*np.array(range(0,accuracy+1)))
        self.enc_string = enc_string
        self.phenotype = self.get_phenotype()
        
    def get_phenotype(self):
                   
        # abs_unscaled_val = np.array(list(map(int,list(self.enc_string[1:])))) @ self.encoding_vector.T
        # abs_string = list(map(int,list(self.enc_string[1:])))
        abs_val_string = ""
        for iterate, allele in enumerate(self.enc_string[1:]):
            abs_val_string += allele
            if iterate == 0:
                abs_val_string += "."

        abs_val = float(abs_val_string)
        if int(self.enc_string[0]) >= 5:
            real_val = abs_val
        elif int(self.enc_string[0]) < 5:
            real_val = -1 * abs_val

        

        if real_val < self.lower_bound or real_val > self.upper_bound:
            real_val = np.inf
        # print(real_val)

        return real_val


class population(chromosome):
    def __init__(self, population_size, accuracy, bounds, random_seed = 42):
        self.accuracy = accuracy
        self.lower_bound, self.upper_bound = bounds
        self.population_size = population_size
        self.rng = np.random.default_rng(random_seed)
        self.generate_legal_phenotypes()
        self.initial_population = self.initialize_population()
        
    def generate_legal_phenotypes(self):
        legal_phenotypes = self.rng.uniform(self.lower_bound, self.upper_bound, self.population_size)
        return legal_phenotypes

    def phenotype_to_chromosome(self, phenotype):
        enc_string = ""
        if phenotype >= 0:
            enc_string += str(self.rng.integers(low = 5, high = 10, size = 1)[0])
        elif phenotype < 0:
            enc_string += str(self.rng.integers(low = 0, high = 5, size = 1)[0])

        coded_string = str(abs(phenotype))[0:self.accuracy+2].replace(".","")
        enc_string += coded_string
       
        return enc_string


    def initialize_population(self):
        legal_phenotypes = self.generate_legal_phenotypes()
        initial_population = [chromosome(self.phenotype_to_chromosome(iterate) ,self.accuracy, (self.lower_bound, self.upper_bound)) for iterate in legal_phenotypes]
        return initial_population

    def get_population_chromosomes(self):
        return [iterate.enc_string for iterate in self.initial_population]
    
    def get_population_fitness(self):
        return [iterate.phenotype for iterate in self.initial_population]



accuracy = 5
bounds = (-0.5,1)
a = population(2,accuracy, bounds, 1)
print(a.get_population_fitness())
print(a.get_population_chromosomes())

