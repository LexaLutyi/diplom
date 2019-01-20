
""" DDE with parameters. """

import matplotlib.pyplot as plt
#from scipy.integrate import odeint
from numpy import *
#from pylab import *
from ddeint import ddeint
from my_model import model_second_order_with_delay, model_second_order_with_delay_and_fixed_graph


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
t_end = 20
# шаг по времени
t_delta = 0.01
# время задержки
t_delay = 2
# сетка значений времени
tt = arange(t_start, t_end, t_delta)
# радиус видимости агентов
vision_radius = 100

# начальные условия в системе выбираем случайными
x0 = random.randn(n_agents * n_axis * n_dim)

x = ddeint(model_second_order_with_delay_and_fixed_graph, lambda t: x0, tt, fargs=(vision_radius, t_delay, n_agents))

# представляем решение в удобном виде
x = x.reshape(tt.size, n_dim, n_axis, n_agents)

plt.figure()
# словарь названий осей
label_dict = {
    (0, 0): 'x',
    (1, 0): 'x\'',
    (0, 1): 'y',
    (1, 1): 'y\'',
    (0, 2): 'z',
    (1, 2): 'z\''
}

for i_dim in range(n_dim):
    for i_axis in range(n_axis):
        # график агентов вдоль одной оси
        plt.plot(tt[:], x[:, i_dim, i_axis, :])
        plt.xlabel('t')
        plt.ylabel(label_dict[(i_dim, i_axis)])
        plt.show()

# график агентов на фазовой плоскости
"""
plt.plot(x[:, 0, 0, :], x[:, 0, 1, :])
plt.xlabel(label_dict[(0, 0)])
plt.ylabel(label_dict[(0, 1)])
plt.show()
"""
y = sum(x[:, 1, :, :], axis=(2,)) / n_agents
plt.plot(tt, y)
plt.xlabel('t')
plt.ylabel('Средняя скорость')
plt.legend(['x', 'y'])
plt.show()

y = y.reshape(tt.size, n_axis, 1)
q = einsum('...k, ...k', x[:, 1, :, :] - y, x[:, 1, :, :] - y) / n_agents
plt.xlabel('t')
plt.ylabel('Среднеквадратичное отклонение от средней скорости')
plt.plot(tt, q)
plt.legend(['x', 'y'])
plt.show()

q = einsum('...k, ...k', x[:, 1, :, 1:-1] - x[:, 1, :, 0:-2], x[:, 1, :, 1:-1] - x[:, 1, :, 0:-2]) / n_agents
plt.xlabel('t')
plt.ylabel('Изменение скорости')
plt.plot(tt, q)
plt.legend(['x', 'y'])
plt.show()