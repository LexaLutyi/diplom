from numpy import *
import matplotlib.pyplot as plt
from scipy.integrate import *
from mpl_toolkits.mplot3d import Axes3D

""""""
def f(y, t):
    dynamic = array([[-5, 1, 1, 1, 1, 1],
                     [1, -5, 1, 1, 1, 1],
                     [1, 1, -5, 1, 1, 1],
                     [1, 1, 1, -5, 1, 1],
                     [1, 1, 1, 1, -5, 1],
                     [1, 1, 1, 1, 1, -5]])
    # print(dynamic @ y)
    return dynamic @ y
""""""
"""
def f(y, t):
    dynamic = array([[-2, 0, 1, 0, 1, 0],
                     [0, -2, 0, 1, 0, 1],
                     [1, 0, -2, 0, 1, 0],
                     [0, 1, 0, -2, 0, 1],
                     [1, 0, 1, 0, -2, 0],
                     [0, 1, 0, 0, 1, -2]])
    # print(dynamic @ y)
    return dynamic @ y
"""

t = arange(0, 20, 0.01)
y0 = [100, 50, 24, 16, -10, -200]
y = odeint(f, y0, t)

# y = array(y).flatten()

plt.figure()
plt.plot(t, [x[0] for x in y],
         t, [x[1] for x in y],
         t, [x[2] for x in y],
         t, [x[3] for x in y],
         t, [x[4] for x in y],
         t, [x[5] for x in y])
plt.grid(True)
plt.show()

"""
plt.figure()
plt.plot([x[0] for x in y], [x[1] for x in y],
         [x[2] for x in y], [x[3] for x in y],
         [x[4] for x in y], [x[5] for x in y])
plt.grid(True)
plt.axis('equal')
plt.show()
"""
"""
plt.figure()
plt.scatter([x[0] for x in y], [x[1] for x in y])
plt.grid(True)
plt.show()
"""

"""
fig = plt.figure()
ax = Axes3D(fig)
ax.plot(t, [x[0] for x in y], [x[1] for x in y])
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('$\dot{x}$')
plt.show()
"""