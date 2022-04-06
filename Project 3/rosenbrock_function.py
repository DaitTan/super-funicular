from tqdm import tqdm
from backbone import Population, BestPositiontracker, PSO
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import numpy as np

def rosenbrock(X):
    X1 = X[:,0]
    X2 = X[:,1]
    "Objective function"
    return 100 * (X2 - X1**2)**2 + (1 - X1)**2


def rosenbrock_cp(x,y):
    "Objective function"
    return 100 * (y - x**2)**2 + (1 - x)**2

pop_size = 40
dim = 2
obj_function = rosenbrock
bounds = [[-5,5], [-5,5]]
init_pop_seed = 1234
pso_seed = 1234

params = {'pop_size' : pop_size,
           'dim' : dim,
           'init_pop_seed' : init_pop_seed,
           'w' : 0.8,
           'c1' :0.1,
           'c2' : 0.1,
           'pso_seed':pso_seed,
           'bounds' : bounds,
           'obj_function' : obj_function}


pop = Population(params)
bps = BestPositiontracker(pop.pop, pop.pop_obj_values)

for x in tqdm(range(3000)):
    pso = PSO(params)
    new_pop = pso(pop, bps, params)
    pop = new_pop
    params['pso_seed'] += 1

x, y = np.array(
            np.meshgrid(
                np.linspace(
                    params['bounds'][0][0],
                    params['bounds'][0][1],
                    1000
                ), 
                np.linspace(
                    params['bounds'][1][0],
                    params['bounds'][1][1],
                    1000
                )
            )
        )

z = rosenbrock_cp(x, y)
 
# Find the global minimum
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]




fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
img = ax.imshow(z, extent=[params['bounds'][0][0],
                            params['bounds'][0][1],
                            params['bounds'][1][0],
                            params['bounds'][1][1]], 
                origin='lower', cmap='viridis', alpha=0.5)

fig.colorbar(img, ax=ax)
ax.plot([1], [1], marker='x', markersize=5, color="white")
contours = ax.contour(x, y, z, 30, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")


pbest_plot = ax.scatter(bps.pbest_sol_vec[:,0], bps.pbest_sol_vec[:,1], marker='o', color='black', alpha=0.5)
# p_plot = ax.scatter(pop.pop[:,0], pop.pop[:,1], marker='o', color='blue', alpha=0.5)
# p_arrow = ax.quiver(pop.pop[:,0], pop.pop[:,1], pop.vel[:,0], pop.vel[:,1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
gbest_plot = plt.scatter([bps.g_best_sol_vec[0]], [bps.g_best_sol_vec[1]], marker='*', s=100, color='black', alpha=0.4)
ax.set_xlim([params['bounds'][0][0], params['bounds'][0][1]])
ax.set_ylim([params['bounds'][1][0], params['bounds'][1][1]])

plt.show()