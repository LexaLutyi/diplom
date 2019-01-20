
""" DDE with parameters. """

import matplotlib.pyplot as plt
#from scipy.integrate import odeint
from numpy import *
#from pylab import *
from ddeint import ddeint
from my_model import model_second_order_with_delay


# фиксируем случайность
seed = random.randint(1, 100000000)
print('seed =', seed)

# число агентов
n_agents = 10
# размерность пространства
n_axis = 2
# порядок системы
n_dim = 2

# время начала моделирования
t_start = 0
# время конца моделирования
t_end = 5
# шаг по времени
t_delta = 0.01
# время задержки
t_delay = 0.1
# сетка значений времени
tt = arange(t_start, t_end, t_delta)

# начальные условия в системе выбираем случайными
x0 = random.randn(n_agents * n_axis * n_dim)

yy = ddeint(model_second_order_with_delay, lambda t: x0, tt, fargs=(10, t_delay, n_agents))

plt.plot(tt, yy)

plt.show()