from sys import flags
import numpy as np
from numpy.random import default_rng
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import image
import cv2
seed = 123458
rng = default_rng(seed)

populationSize = 100

# x1 y1 x2 y2
bounds = [[-10,10],[-10,10],[-10,10],[-10,10]]

population = []
for b in bounds:
    var = b[0] + (rng.random((populationSize, )) * (b[1]-b[0]))
    population.append(var)

population = np.array(population).T
print(population.shape)


im = cv2.imread("sher.jpg", flags = 0)
sobelxy = cv2.Sobel(src=im, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

# cv2.imshow("", im)
# cv2.waitKey(0)
for y in population:
    plt.plot([y[0], y[2]], [y[1], y[3]])

# plt.xlim((0, im.shape[0]))
# plt.ylim((0, im.shape[1]))
plt.axis('off')
plt.show()
plt.savefig("test.png", bbox_inches='tight')

# imx = cv2.imread("test.png", flags = 0)
# print(im.shape)