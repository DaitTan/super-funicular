from cProfile import label
from turtle import color
import numpy as np
import pickle
import matplotlib.pyplot as plt



def generate_plot(result_changin, c, str_xlabel, save_as):
    mean_min = np.mean(np.min(result_changin, 1), 1)
    mean_min_std = np.std(np.min(result_changin, 1), 1)
    min_ci = 1.96 * mean_min_std/np.sqrt(result_changin.shape[2])

    mean_avg = np.mean(np.mean(result_changin, 1), 1)
    mean_avg_std = np.std(np.mean(result_changin, 1), 1)
    avg_ci = 1.96 * mean_avg_std/np.sqrt(result_changin.shape[2])

    mean_max = np.mean(np.max(result_changin, 1), 1)
    mean_max_std = np.std(np.max(result_changin, 1), 1)
    max_ci = 1.96 * mean_max_std/np.sqrt(result_changin.shape[2])


    fig, ax = plt.subplots()
    
    ax.plot(c,mean_min, ".-", color = "r", label = "Mean Min")
    ax.fill_between(c,(mean_min - min_ci),(mean_min + min_ci), color='r', alpha=.1)

    ax.plot(c,mean_avg, ".-", color = "b", label = "Mean Avg")
    ax.fill_between(c,(mean_avg - avg_ci),(mean_avg + avg_ci), color='b', alpha=.1)

    ax.plot(c,mean_max, ".-", color = "g", label = "Mean Max")
    ax.fill_between(c,(mean_max - max_ci),(mean_max + max_ci), color='g', alpha=.1)
    ax.set_xlabel(str_xlabel)
    ax.set_ylabel("Objective Function")
    plt.legend()
    fig.tight_layout()
    plt.savefig(save_as, dpi = 500, format = 'png')


with open("himmelblaus_changing_c1.pkl", "rb") as f:
    result_changin_c1 = pickle.load(f)

c1 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
generate_plot(result_changin_c1, c1, "c1", "himmelblaus_changing_c1.png")

with open("himmelblaus_changing_c2.pkl", "rb") as f:
    result_changin_c2 = pickle.load(f)

c2 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
generate_plot(result_changin_c2, c2, "c2", "himmelblaus_changing_c2.png")

with open("himmelblaus_changing_w.pkl", "rb") as f:
    result_changin_w = pickle.load(f)

w = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
generate_plot(result_changin_w, w, "w", "himmelblaus_changing_w.png")