
""" DDE with parameters. """

from pylab import *
from ddeint import ddeint


def model(Y, t):
    x1, x2 = Y(t-1)
    x = array([-x1, -x2])
    return x


tt = linspace(0, 30, 100)
yy = ddeint(model, lambda t:array([1,2]), tt)

fig, ax = subplots(1,figsize=(4,4))
ax.plot(tt, yy)

show()