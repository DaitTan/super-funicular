from tqdm import tqdm
from backbone import Population, BestPositiontracker, PSO
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
import numpy as np

def three_hump_camel(X):
    X1 = X[:,0]
    X2 = X[:,1]
    "Objective function"
    return (2 * X1**2) - (1.05 * X1**4) + ((X1**6)/6.) + (X1*X2) + (X2**2)


def three_hump_camel_cp(x,y):
    "Objective function"
    return (2 * x**2) - (1.05 * x**4) + ((x**6)/6.) + (x*y) + (y**2)

#############################################################################
# pop_size = 100
# dim = 2
# obj_function = three_hump_camel
# bounds = [[-3,3], [-3,3]]
# init_pop_seed = 1234
# pso_seed = 1234

# params = {'pop_size' : pop_size,
#            'dim' : dim,
#            'init_pop_seed' : init_pop_seed,
#            'w' : 0.8,
#            'c2' : 0.1,
#            'pso_seed':pso_seed,
#            'bounds' : bounds,
#            'obj_function' : obj_function}

# c1 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# total_c1 = len(c1)
# total_pop_size = pop_size
# replications = 10
# generations = 10000
# result_arr = np.zeros((total_c1,pop_size,replications))
# pbest_track = np.zeros((total_c1,replications, pop_size, dim))
# for c in tqdm(range(len(c1))):
#     params['c1'] = c1[c]
    
#     for seed_add in tqdm(range(replications)):
#         params['init_pop_seed'] = 12345 + seed_add 
#         params['pso_seed'] = 123456 + seed_add 
#         pop = Population(params)
#         bps = BestPositiontracker(pop.pop, pop.pop_obj_values)
#         # print(params)
        
#         for x in range(generations):
#             pso = PSO(params)
#             new_pop = pso(pop, bps, params)
#             pop = new_pop
#             params['pso_seed'] += 1

#         pbest_track[c,seed_add,:,:] = bps.g_best_sol_vec
#         result_arr[c,:,seed_add] = bps.pbest_sol

# import pickle 
# with open("hc3_data/hc3_changing_c1_output.pkl","wb") as f:
#     pickle.dump(result_arr, f)

# with open("hc3_data/hc3_changing_c1_input.pkl","wb") as f:
#     pickle.dump(pbest_track, f)
#############################################################################
# pop_size = 100
# dim = 2
# obj_function = three_hump_camel
# bounds = [[-3,3], [-3,3]]
# init_pop_seed = 1234
# pso_seed = 1234

# params = {'pop_size' : pop_size,
#            'dim' : dim,
#            'init_pop_seed' : init_pop_seed,
#            'w' : 0.8,
#            'c1' : 0.1,
#            'pso_seed':pso_seed,
#            'bounds' : bounds,
#            'obj_function' : obj_function}

# c2 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# total_c2 = len(c2)
# total_pop_size = pop_size
# replications = 10
# generations = 10000
# result_arr = np.zeros((total_c1,pop_size,replications))
# pbest_track = np.zeros((total_c1,replications, pop_size, dim))
# for c in tqdm(range(len(c2))):
#     params['c2'] = c2[c]
    
#     for seed_add in tqdm(range(replications)):
#         params['init_pop_seed'] = 12345 + seed_add 
#         params['pso_seed'] = 123456 + seed_add 
#         pop = Population(params)
#         bps = BestPositiontracker(pop.pop, pop.pop_obj_values)
#         # print(params)
        
#         for x in range(generations):
#             pso = PSO(params)
#             new_pop = pso(pop, bps, params)
#             pop = new_pop
#             params['pso_seed'] += 1
#         pbest_track[c,seed_add,:,:] = bps.g_best_sol_vec
#         result_arr[c,:,seed_add] = bps.pbest_sol

# import pickle 
# with open("hc3_data/hc3_changing_c2_output.pkl","wb") as f:
#     pickle.dump(result_arr, f)

# with open("hc3_data/hc3_changing_c2_input.pkl","wb") as f:
#     pickle.dump(pbest_track, f)
# #############################################################################
# pop_size = 100
# dim = 2
# obj_function = three_hump_camel
# bounds = [[-3,3], [-3,3]]
# init_pop_seed = 1234
# pso_seed = 1234

# params = {'pop_size' : pop_size,
#            'dim' : dim,
#            'init_pop_seed' : init_pop_seed,
#            'c1' : 0.1,
#            'c2' : 0.1,
#            'pso_seed':pso_seed,
#            'bounds' : bounds,
#            'obj_function' : obj_function}

# w = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# total_w = len(w)
# total_pop_size = pop_size
# replications = 10
# generations = 10000
# result_arr = np.zeros((total_c1,pop_size,replications))
# pbest_track = np.zeros((total_c1,replications, pop_size, dim))
# for c in tqdm(range(len(w))):
#     params['w'] = w[c]
    
#     for seed_add in tqdm(range(replications)):
#         params['init_pop_seed'] = 12345 + seed_add 
#         params['pso_seed'] = 123456 + seed_add 
#         pop = Population(params)
#         bps = BestPositiontracker(pop.pop, pop.pop_obj_values)
#         # print(params)
        
#         for x in range(generations):
#             pso = PSO(params)
#             new_pop = pso(pop, bps, params)
#             pop = new_pop
#             params['pso_seed'] += 1
#         pbest_track[c,seed_add,:,:] = bps.g_best_sol_vec
#         result_arr[c,:,seed_add] = bps.pbest_sol

# import pickle 
# with open("hc3_data/hc3_changing_w_output.pkl","wb") as f:
#     pickle.dump(result_arr, f)

# with open("hc3_data/hc3_changing_w_input.pkl","wb") as f:
#     pickle.dump(pbest_track, f)
    

pop_size = 100
dim = 2
obj_function = three_hump_camel
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
pso = PSO(params)

replications = 20
generation = 10000
result_arr = np.zeros((generation, pop_size, replications))

for x in range(replications):
    params['init_pop_seed'] += x
    params['pso_seed'] += x
    for y in tqdm(range(generation)):
        new_pop = pso(pop, bps, params)
        pop = new_pop
        result_arr[y,:,x] = bps.pbest_sol

print(np.mean(np.mean(result_arr,1),1))

fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
ax.plot(np.mean(np.min(result_arr,1),1), label="Min of Pop")
ax.plot(np.mean(np.max(result_arr,1),1), label="Mean of Pop")
ax.plot(np.mean(np.mean(result_arr,1),1), label="Max of Pop")

ax.set_xlabel("Generations")
ax.set_ylabel("Objective Function")
plt.legend()
fig.tight_layout()
plt.savefig("HC3_convergence.png", dpi = 500, format = 'png')



# plt.show()
x, y = np.array(
            np.meshgrid(
                np.linspace(
                    params['bounds'][0][0],
                    params['bounds'][0][1],
                    100
                ), 
                np.linspace(
                    params['bounds'][1][0],
                    params['bounds'][1][1],
                    100
                )
            )
        )

z = three_hump_camel_cp(x, y)
 
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

contours = ax.contour(x, y, z, 100, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")


pbest_plot = ax.scatter(bps.pbest_sol_vec[:,0], bps.pbest_sol_vec[:,1], marker='o', color='black', alpha=0.5, label="Particles")
# p_plot = ax.scatter(pop.pop[:,0], pop.pop[:,1], marker='o', color='blue', alpha=0.5)
# p_arrow = ax.quiver(pop.pop[:,0], pop.pop[:,1], pop.vel[:,0], pop.vel[:,1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
gbest_plot = plt.scatter([bps.g_best_sol_vec[0]], [bps.g_best_sol_vec[1]], marker='*', s=100, color='black', alpha=0.4, label="Global Minimums Found")
ax.scatter([0], [0], marker='x', color="white", label="Global Minimums")
ax.set_xlim([params['bounds'][0][0], params['bounds'][0][1]])
ax.set_ylim([params['bounds'][1][0], params['bounds'][1][1]])


ax.set_xlabel("x")
ax.set_ylabel("y")
plt.legend()
fig.tight_layout()
plt.savefig("HC3_contour.png", dpi = 500, format = 'png')

print(f"{np.mean(np.min(result_arr,1),1)[-1]} \t {np.mean(np.mean(result_arr,1),1)[-1]} \t {np.mean(np.max(result_arr,1),1)[-1]}")