import pickle
import numpy as np

with open("data/question_1.pickle", "rb") as output_file:
    x_history_q1, y_history_q1, time_q1 = pickle.load(output_file)

with open("data/question_2_noise0_5.pickle", "rb") as output_file:
    x_history_q2_1, y_history_q2_1, time_q2_1 = pickle.load(output_file)

with open("data/question_2_noise1.pickle", "rb") as output_file:
    x_history_q2_2, y_history_q2_2, time_q2_2 = pickle.load(output_file)

def objective_function(x):
    return (x * np.sin(10*np.pi*x) + 1)


mean_pop_over_gen_q1 = [np.round(np.mean(y_history_q1[i,-1,:]),5) for i in range(y_history_q1.shape[0])]
min_pop_over_gen_q1 = [np.round(np.min(y_history_q1[i,-1,:]),5) for i in range(y_history_q1.shape[0])]
max_pop_over_gen_q1 = [np.round(np.max(y_history_q1[i,-1,:]),5) for i in range(y_history_q1.shape[0])]

mean_pop_over_gen_q2_1 = [np.round(np.mean(y_history_q2_1[i,-1,:]),5) for i in range(y_history_q2_1.shape[0])]
min_pop_over_gen_q2_1 = [np.round(np.min(y_history_q2_1[i,-1,:]),5) for i in range(y_history_q2_1.shape[0])]
max_pop_over_gen_q2_1 = [np.round(np.max(y_history_q2_1[i,-1,:]),5) for i in range(y_history_q2_1.shape[0])]


mean_pop_over_gen_q2_2 = [np.round(np.mean(y_history_q2_2[i,-1,:]),5) for i in range(y_history_q2_2.shape[0])]
min_pop_over_gen_q2_2 = [np.round(np.min(y_history_q2_2[i,-1,:]),5) for i in range(y_history_q2_2.shape[0])]
max_pop_over_gen_q2_2 = [np.round(np.max(y_history_q2_2[i,-1,:]),5) for i in range(y_history_q2_2.shape[0])]

result = np.zeros((13,3))
result[0,0], result[0,1], result[0,2] = (np.mean(time_q1), np.mean(time_q2_1), np.mean(time_q2_2))
result[1,0], result[1,1], result[1,2] = (np.mean(max_pop_over_gen_q1), np.mean(max_pop_over_gen_q2_1),np.mean(max_pop_over_gen_q2_2))
result[2,0], result[2,1], result[2,2] = (np.var(max_pop_over_gen_q1), np.var(max_pop_over_gen_q2_1), np.var(max_pop_over_gen_q2_2))
result[3,0], result[3,1], result[3,2] = (np.min(max_pop_over_gen_q1), np.min(max_pop_over_gen_q2_1), np.min(max_pop_over_gen_q2_2))
result[4,0], result[4,1], result[4,2] = (np.max(max_pop_over_gen_q1), np.max(max_pop_over_gen_q2_1), np.max(max_pop_over_gen_q2_2))

result[5,0], result[5,1], result[5,2] = (np.mean(mean_pop_over_gen_q1), np.mean(mean_pop_over_gen_q2_1), np.mean(mean_pop_over_gen_q2_2))
result[6,0], result[6,1], result[6,2] = (np.var(mean_pop_over_gen_q1), np.var(mean_pop_over_gen_q2_1), np.var(mean_pop_over_gen_q2_2))
result[7,0], result[7,1], result[7,2] = (np.min(mean_pop_over_gen_q1), np.min(mean_pop_over_gen_q2_1), np.min(mean_pop_over_gen_q2_2))
result[8,0], result[8,1], result[8,2] = (np.max(mean_pop_over_gen_q1), np.max(mean_pop_over_gen_q2_1), np.max(mean_pop_over_gen_q2_2))


result[9,0], result[9,1], result[9,2] = (np.mean(min_pop_over_gen_q1), np.mean(min_pop_over_gen_q2_1), np.mean(min_pop_over_gen_q2_2))
result[10,0], result[10,1], result[10,2] = (np.var(min_pop_over_gen_q1), np.var(min_pop_over_gen_q2_1), np.var(min_pop_over_gen_q2_2))
result[11,0], result[12,1], result[11,2] = (np.min(min_pop_over_gen_q1), np.min(min_pop_over_gen_q2_1), np.min(min_pop_over_gen_q2_2))
result[12,0], result[12,1], result[12,2] = (np.max(min_pop_over_gen_q1), np.max(min_pop_over_gen_q2_1), np.max(min_pop_over_gen_q2_2))


import pandas as pd
pd.set_option('display.float_format', '{:.5E}'.format)
df = pd.DataFrame(result, columns = ['q1','q2','q3'])
print(df.to_latex(index=False))



# print(mean_pop_over_gen_q1)
# print(mean_pop_over_gen_q2)
from matplotlib import animation

import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(ylim=(0,2))
# ax = ax.axes() 
ax.plot(max_pop_over_gen_q1, "-", label = "Max. Fitness using B-10E")
ax.plot(max_pop_over_gen_q2_1, "-", label = "Max. Fitness using CE N(0,0.05)")
ax.plot(max_pop_over_gen_q2_2, "-", label = "Max. Fitness using CE N(0,1)")
ax.plot(max_pop_over_gen_q1, ".", markersize = 3)
ax.plot(max_pop_over_gen_q2_1, ".", markersize = 3)
ax.plot(max_pop_over_gen_q2_2, ".", markersize = 3)
ax.set_xlabel("Runs")
ax.set_ylabel("Best Fitness from Final Pop.")
  
ax.legend()
fig.tight_layout()
plt.savefig("images/q2_maxplot.png", dpi = 1000, format = 'png')


fig = plt.figure()
ax = plt.axes(ylim=(0,2))
# ax.axes(ylim=(0,1)) 
ax.plot(mean_pop_over_gen_q1, "-", label = "Mean Fitness using B-10E")
ax.plot(mean_pop_over_gen_q2_1, "-", label = "Mean Fitness using CE N(0,0.05)")
ax.plot(mean_pop_over_gen_q2_2, "-", label = "Mean Fitness using CE N(0,1)")
ax.plot(mean_pop_over_gen_q1, ".", markersize = 3)
ax.plot(mean_pop_over_gen_q2_1, ".", markersize = 3)
ax.plot(mean_pop_over_gen_q2_2, ".", markersize = 3)
ax.set_xlabel("Runs")
ax.set_ylabel("Mean Fitness of Final Pop.")
ax.legend()
fig.tight_layout()
plt.savefig("images/q2_meanplot.png", dpi = 1000, format = 'png')

fig = plt.figure()
ax = plt.axes(ylim=(0,2))
# ax.axes(ylim=(0,1)) 
ax.plot(min_pop_over_gen_q1, "-", label = "Min. Fitness using B-10E")
ax.plot(min_pop_over_gen_q2_1, "-", label = "Min. Fitness using CE N(0,0.05)")
ax.plot(min_pop_over_gen_q2_2, "-", label = "Min. Fitness using CE N(0,1)")
ax.plot(min_pop_over_gen_q1, ".", markersize = 3)
ax.plot(min_pop_over_gen_q2_1, ".", markersize = 3)
ax.plot(min_pop_over_gen_q2_1, ".", markersize = 3)
ax.set_xlabel("Runs")
ax.set_ylabel("Min. Fitness of Final Pop.")
ax.legend()
fig.tight_layout()
plt.savefig("images/q2_minplot.png", dpi = 1000, format = 'png')

# plt.show()

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = plt.axes()
# ax.plot(np.mean(y_history_q1[10,:,:],1), linewidth = 0.5, label = "Maximum Fitness using base-10 encoding")
# ax.plot(np.mean(y_history_q2[10,:,:],1), linewidth = 0.5, label = "Maximum Fitness using continuous encoding")
# ax.set_xlabel("Number of Generations")
# ax.set_ylabel("Fitness")
    
# ax.legend()
# fig.tight_layout()
# plt.savefig("images/min_max_mean_q1q2.png", dpi = 1000, format = 'png')
# # plt.show()


# fig = plt.figure()
# ax = plt.axes()

# x_best_q1 = []
# x_best_q2 = []
# for iter, y in enumerate(y_history_q1[0,:,:]):
#     ind = np.argmax(y)
#     x_best_q1.append(x_history_q1[10,iter,ind])
#     x_best_q2.append(x_history_q2[10,iter,ind])

# ax.plot(x_best_q1, label = "Trajectory of best point - Base 10 Encoding")
# # ax.plot(x_best_q1, "ok", markersize = 1, label = "Points base 10 Encoding")
# ax.plot(x_best_q2, label = "Trajectory of best point - Cont. Encoding")
# # ax.plot(x_best_q2, "or", markersize = 1, label = "Points Cont. Encoding")
# ax.set_xlabel("Number of Generation")
# ax.set_ylabel("Best Point")
# ax.legend()
# fig.tight_layout()
# plt.savefig("images/point_trajectory_q1q2.png", dpi = 500, format = 'png')