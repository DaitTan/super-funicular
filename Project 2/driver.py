import enum
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import itertools
import pandas as pd

class Objectives:
    def __init__(self, objectives):
        self.signs = np.ones((len(objectives)))
        for iterate in range(len(objectives)):
            if objectives[iterate]["type"] == "Maximize":
                self.signs[iterate] = -1

        self.objectives = objectives

    def evaluate(self, population):
        m, _ = population.shape
        sol = np.zeros((m, len(self.objectives)))

        for iterate in range(len(self.objectives)):
            sol[:, iterate] = self.signs[iterate] * self.objectives[iterate]["function"](population)
        
        return sol


class Frontiers:
    class_id = itertools.count(1,1)
    def __init__(self):
        self.frontier_id = next(self.class_id)
        print("This id is {}".format(self.frontier_id))
        self.points_in_frontier = []
        

    def add(self, serial_number):
        self.points_in_frontier.append(serial_number)

class NonDominatedSorting:
    def __init__(self, population):
        self.points_frontier_class = []
        frontier = Frontiers()
        
        for iterate_1 in range(len(population.population)):
            for iterate_2 in range(len(population.population)):
                
                if self.dominates(population.population[iterate_1].corres_eval, 
                                    population.population[iterate_2].corres_eval):

                    population.population[iterate_1].S.add(
                                            population.population[iterate_2].serial_number
                                            )
                                           
                elif self.dominates(population.population[iterate_2].corres_eval, 
                                    population.population[iterate_1].corres_eval):
                    
                    population.population[iterate_1].n += 1

            if population.population[iterate_1].n == 0:
                population.population[iterate_1].rank = 1
                frontier.add(population.population[iterate_1].serial_number)

        all_frontiers = [frontier]
        iterate = 0
        while not len(all_frontiers[iterate].points_in_frontier)==0:
            Q = []
            for p_id in all_frontiers[iterate].points_in_frontier:
                point_p, index_p = population.fetch_by_serial_number(p_id)
                for q_id in list(point_p.S):
                    point_q, index_q = population.fetch_by_serial_number(q_id)
                    population.population[index_q].n -= 1
                    if population.population[index_q].n == 0:
                        print(population.population[index_q].rank)
                        population.population[index_q].rank = iterate + 2
                        print(population.population[index_q].rank)
                        Q.append(q_id)
            
            for x in all_frontiers[iterate].points_in_frontier:
                _, index = population.fetch_by_serial_number(x)
                
                population.population[index].frontier = all_frontiers[iterate].frontier_id

            iterate += 1
            # print("Length Q = {}".format(len(Q)))
            
            next_frontier = Frontiers()
            for id in Q:
                next_frontier.add(id)
            
            all_frontiers.append(next_frontier)

        self.all_frontiers = all_frontiers[:-1]

    def crowding_distance(self, population):
        for _, frontiers in enumerate(self.all_frontiers):
            cardinality_r = len(frontiers.points_in_frontier)
            evaluations = []
            sr_num = []
            for iterate in frontiers.points_in_frontier:
                point,_ = population.fetch_by_serial_number(iterate)
                evaluations.append(point.corres_eval)
                sr_num.append(point.serial_number)
            evaluations = np.array(evaluations)
            sr_num = np.array(sr_num)
            for objective_num in range(population.num_objectives):
                sort_indices = np.argsort(evaluations[:,objective_num])
                print(sort_indices)
                evaluations = evaluations[sort_indices, :]
                sr_num = sr_num[sort_indices]
                print(sr_num)
                _, first_index = population.fetch_by_serial_number(sr_num[0])
                _, last_index = population.fetch_by_serial_number(sr_num[-1])
                population.population[first_index].d = float("inf")
                population.population[last_index].d = float("inf")

                for i in range(1, cardinality_r-1):
                    _, index = population.fetch_by_serial_number(sr_num[i])
                    population.population[index].d += (abs(
                                            evaluations[i+1,objective_num] - evaluations[i-1,objective_num]
                                        )) / (evaluations[-1, objective_num] - evaluations[0, objective_num])

    def dominates(self, p, q):
        comparison = p < q
        # print(comparison)
        # print(comparison)
        # print(np.all(comparison))
        # print("*******")
        if np.all(comparison):
            return True
        else:
            return False

class SolutionVecProps:
    class_id = itertools.count()
    def __init__(self, sol_vec, corres_eval):
        self.serial_number = next(self.class_id)
        self.rank = 0
        self.sol_vec = sol_vec
        self.corres_eval = corres_eval
        self.S = set()
        self.n = 0
        self.d = 0
        self.frontier = -1



class Population:
    def __init__(self, population_size, num_variables, bounds, 
                    objectives, defined_pop = [], generate = True):

        self.population_size = population_size
        self.num_objectives = len(objectives.objectives)
        self.num_variables = num_variables
        self.bounds = bounds
        self.objectives = objectives
        if generate == True:
            # sol_vectors = self.generate_random_legal_population()
            sol_vectors = np.array([[0.913, 2.181],
                                    [0.599, 2.450],
                                    [0.139, 1.157],
                                    [0.867, 1.505],
                                    [0.885, 1.239],
                                    [0.658, 2.040],
                                    [0.788, 2.166],
                                    [0.342, 0.756]])

            evaluations = self.evaluate_objectives(sol_vectors)
            self.population = self.generate_population(sol_vectors, evaluations)
        else:
            self.population = defined_pop


    def get_all_sol_vecs(self):
        return np.array([iterate.sol_vec for iterate in self.population])

    def get_all_evals(self):
        return np.array([iterate.corres_eval for iterate in self.population])

    def get_all_serial_numbers(self):
        return [iterate.serial_number for iterate in self.population]

    def generate_population(self, sol_vectors, evaluations):
        population = []
        for sol_vec, corres_eval in zip(sol_vectors, evaluations):
            pointProp = SolutionVecProps(sol_vec, corres_eval)
            population.append(pointProp)
        return population

    def fetch_by_serial_number(self, target):
        for iterate, pop in enumerate(self.population):
            if pop.serial_number == target:
                return pop, iterate
        

    def generate_random_legal_population(self):
        population = np.random.rand(self.population_size, self.num_variables)
        for iterate in range(self.num_variables):
            lower_b = self.bounds[iterate][0]
            upper_b = self.bounds[iterate][1]
            population[:, iterate] = (population[:, iterate] * (upper_b - lower_b)) + lower_b

        return population

    def plotPopulation(self):
        fig = go.Figure()
        evaluations = self.get_all_evals()
        point_caption = (["Point {}".format(i) for i in self.get_all_serial_numbers()])
        fig.add_trace(go.Scatter(
            x = evaluations[:,0],
            y = evaluations[:,1],
            mode = "markers",
            text = point_caption
        ))
        
        fig.show()
    
    def plotPopulationwithFrontier(self):
        fig = go.Figure()
        all_frontiers = NonDominatedSorting(self)
        for rank, frontiers in enumerate(all_frontiers.all_frontiers):
            evaluations = []
            sr_num = []
            for iterate in frontiers.points_in_frontier:
                point,_ = self.fetch_by_serial_number(iterate)
                evaluations.append(point.corres_eval)
                sr_num.append(point.serial_number)
            evaluations = np.array(evaluations)
            df = pd.DataFrame(dict(
                    x = evaluations[:,0],
                    y = evaluations[:,1],
                ))
            point_caption = (["Point {}".format(i) for i in sr_num])
            fig.add_trace(go.Scatter(
            x = df.sort_values(by="x")["x"],
            y = df.sort_values(by="x")["y"],
            mode = "markers+lines",
            text = point_caption,
            name = "Frontier {}".format(rank + 1)
            ))
        fig.show()

    def evaluate_objectives(self, sol_vectors):
        # population = self.population
        return self.objectives.evaluate(sol_vectors)

    



def obj_1(pop):
    x = pop[:, 0]
    return x

def obj_2(pop):
    x = pop[:,0]
    y = pop[:,1]
    return 1 + y - x**2

objective_1 = {}
objective_1["type"] = "Minimize"
objective_1["function"] = obj_1

objective_2 = {}
objective_2["type"] = "Minimize"
objective_2["function"] = obj_2

objectives_list = [objective_1, objective_2]

objectives = Objectives(objectives_list)

            
pop = Population(8, 2, [[0,1],[0,3]], objectives)

pop2 = Population(10, 2, [[0,1],[0,3]], objectives)
print([iterate.serial_number for iterate in pop.population])
print([iterate.serial_number for iterate in pop2.population])

# print(pop.evaluations)
# fds = NonDominatedSorting(pop)
# fds.crowding_distance(pop)

# print([iterate.frontier for iterate in pop.population])
# print([iterate.rank for iterate in pop.population])
# print([iterate.d for iterate in pop.population])
# print(pop.get_all_sol_vecs())
# print(pop.get_all_evals())

# pop.plotPopulationwithFrontier()


# print([iterate.serial_number for iterate in pop.population])
# pop.plotPopulation()

