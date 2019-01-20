from numpy import *
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.integrate import *


time = datetime.now()
a = array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4])
b = reshape(a, (12, 1, 1))
print(b)
print(b[0][0][0])
print(time - datetime.now())


def ez_plot(t, flat_y, dim = 1, order = 1):
    y = flat_y.reshape((flat_y % (dim * order), dim, order))
    plt.figure()
    for dim_i in range(dim):
        for order_i in range(order):
            for agent in y:
                plt.plot(t, agent[dim_i][order_i])
            plt.show()
    if dim == 2:
        for agent in y:
            plt.plot(agent[0][0], agent[1][0])
        plt.show()
