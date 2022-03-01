from scipy.optimize import minimize
from numpy.random import rand
import numpy as np
from scipy.optimize import dual_annealing

def objective(x):
    return -1*(x * np.sin(10*np.pi*x) + 1)

 
# define range for input
r_min, r_max = -0.5, 1.0
# define the bounds on the search
bounds = [[r_min, r_max]]
# perform the simulated annealing search
result = dual_annealing(objective, bounds)
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f({}) = {}'.format(solution, evaluation))