import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
class BestPositiontracker:
    def __init__(self, sol_vec, sol) -> None:
        self.pbest_sol_vec = sol_vec
        self.pbest_sol = sol
        g_best_sol_index = np.argmin(self.pbest_sol)
        self.g_best_sol_vec = self.pbest_sol_vec[g_best_sol_index,:]
        self.g_best_sol = self.pbest_sol[g_best_sol_index]

    def update(self, sol_vec, sol):
        self.pbest_sol_vec[(self.pbest_sol >= sol), :] = sol_vec[(self.pbest_sol >= sol), :]
        self.pbest_sol[(self.pbest_sol >= sol)] = sol[(self.pbest_sol >= sol)]
        g_best_sol_index = np.argmin(self.pbest_sol)
        self.g_best_sol_vec = self.pbest_sol_vec[g_best_sol_index,:]
        self.g_best_sol = self.pbest_sol[g_best_sol_index]

class Population:
    def __init__(self, params, pos_vel = None):
        self.pop_size = params['pop_size']
        self.bounds = params['bounds']
        self.dim = params['dim']
        self.rng = default_rng(params["init_pop_seed"])

        if pos_vel is not None:
            pop, vel = pos_vel
            assert pop.shape[0] == self.pop_size
            assert pop.shape[1] == self.dim
            assert vel.shape[0] == self.pop_size
            assert vel.shape[1] == self.dim
            

            self.pop = pop
            self.vel = vel
            
        else:
            self.pop = self.initialize_pop()
            self.vel = self.initialize_vel()
            self.pbest = self.pop

        self.pop_obj_values = params['obj_function'](self.pop)
        

    def initialize_pop(self):
        population = self.rng.random((self.pop_size, self.dim))
        for iterate in range(self.dim):
            lower_b = self.bounds[iterate][0]
            upper_b = self.bounds[iterate][1]
            population[:, iterate] = (population[:, iterate] * (upper_b - lower_b)) + lower_b
    
        return population
    
    def initialize_vel(self):
        velocity = self.rng.random((self.pop_size, self.dim))
        for iterate in range(self.dim):
            # lower_b = self.bounds[iterate][0]
            # upper_b = self.bounds[iterate][1]
            velocity[:, iterate] = velocity[:, iterate]*0.01
    
        return velocity
    

    def evaluate_pop(self):
        return self.obj_fun(self.pop)

class PSO:
    def __init__(self, params):
        self.rng = default_rng(params['pso_seed'])
        self.w = params['w'] 
        self.c1 = params['c1']
        self.c2 = params['c2']

    def __call__(self, population, bps, params):
        new_velo = self.updateVelo(population, bps)
        new_pop = population.pop + new_velo
        new_pop = Population(params, (new_pop, new_velo))
        bps.update(new_pop.pop, new_pop.pop_obj_values)
        return new_pop

    def updateVelo(self, population, bps):
        r1 = self.rng.random()
        r2 = self.rng.random()
        new_velo = ((self.w * population.vel) 
                    + (self.c1 * r1 * (bps.pbest_sol_vec - population.pop)) 
                    + (self.c2 * r2 * (bps.g_best_sol_vec - population.pop)))

        return new_velo


# def f(X):
#     X1 = X[:,0]
#     X2 = X[:,1]
#     "Objective function"
#     return (X1-3.14)**2 + (X2-2.72)**2 + np.sin(3*X1+1.41) + np.sin(4*X2-1.73)

# def rosenbrock(X):
#     X1 = X[:,0]
#     X2 = X[:,1]
#     "Objective function"
#     return 100 * (X2 - X1**2)**2 + (1 - X1)**2

# def beale(X):
#     X1 = X[:,0]
#     X2 = X[:,1]
#     "Objective function"
#     return (1.5 - X1 + X1*X2)**2 + (2.25 - X1 + X1*X2**2)**2 + (2.625 - X1 + X1*X2**3)**2

# pop_size = 20
# dim = 2
# obj_function = beale
# bounds = [[-5,5], [-5,5]]
# init_pop_seed = 1234
# pso_seed = 1234

# params = {'pop_size' : pop_size,
#            'dim' : dim,
#            'init_pop_seed' : init_pop_seed,
#            'w' : 0.8,
#            'c1' :0.1,
#            'c2' : 0.1,
#            'pso_seed':pso_seed,
#            'bounds' : bounds}


# # pop = Population(params)
# # bps = BestPositiontracker(pop.pop, pop.pop_obj_values)
# # print(vars(bps))
# # for x in tqdm(range(1000)):
# #     pso = PSO(params)
# #     new_pop = pso(pop, bps, params)
# #     pop = new_pop
# #     params['pso_seed'] += 1

# def f_cp(x,y):
#     "Objective function"
#     return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

# def beale_cp(x,y):
#     return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
# x, y = np.array(
#             np.meshgrid(
#                 np.linspace(
#                     params['bounds'][0][0],
#                     params['bounds'][0][1],
#                     100
#                 ), 
#                 np.linspace(
#                     params['bounds'][1][0],
#                     params['bounds'][1][1],
#                     100
#                 )
#             )
#         )

# z = beale_cp(x, y)
 
# # Find the global minimum
# x_min = x.ravel()[z.argmin()]
# y_min = y.ravel()[z.argmin()]


# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# fig, ax = plt.subplots(figsize=(8,6))
# fig.set_tight_layout(True)
# img = ax.imshow(z, extent=[params['bounds'][0][0],
#                             params['bounds'][0][1],
#                             params['bounds'][1][0],
#                             params['bounds'][1][1]], 
#                 origin='lower', cmap='viridis', alpha=0.5)

# fig.colorbar(img, ax=ax)
# ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
# contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
# ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")


# pso = PSO(params)
# pop = Population(params)
# bps = BestPositiontracker(pop.pop, pop.pop_obj_values)


# pbest_plot = ax.scatter(bps.pbest_sol_vec[:,0], bps.pbest_sol_vec[:,1], marker='o', color='black', alpha=0.5)
# p_plot = ax.scatter(pop.pop[:,0], pop.pop[:,1], marker='o', color='blue', alpha=0.5)
# p_arrow = ax.quiver(pop.pop[:,0], pop.pop[:,1], pop.vel[:,0], pop.vel[:,1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
# gbest_plot = plt.scatter([bps.g_best_sol_vec[0]], [bps.g_best_sol_vec[1]], marker='*', s=100, color='black', alpha=0.4)
# ax.set_xlim([params['bounds'][0][0], params['bounds'][0][1]])
# ax.set_ylim([params['bounds'][1][0], params['bounds'][1][1]])

# def animate(i):
#     "Steps of PSO: algorithm update and show in plot"
#     title = 'Iteration {:02d}'.format(i)
#     print(i)
#     # Update params
#     global pso
#     global pop
#     global bps
#     new_pop = pso(pop, bps, params)
#     pop = new_pop
#     params['pso_seed'] += 1
#     # Set picture
#     ax.set_title(title)
#     pbest_plot.set_offsets(bps.pbest_sol_vec)
#     p_plot.set_offsets(bps.pbest_sol_vec)
#     p_arrow.set_offsets(bps.pbest_sol_vec)
#     p_arrow.set_UVC(pop.vel[:,0], pop.vel[:,1])
#     gbest_plot.set_offsets(bps.g_best_sol_vec)
#     return ax, pbest_plot, p_plot, p_arrow, gbest_plot

# anim = FuncAnimation(fig, animate, frames=list(range(1,50)), interval=500, blit=False, repeat=True)
# anim.save("PSO.gif", dpi=120, writer="imagemagick")

# print(vars(bps))