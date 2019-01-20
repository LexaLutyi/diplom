import matplotlib.pyplot as plt
from scipy.integrate import odeint
from time import clock
from numpy import *
from my_model import model_second_order

# фиксируем случайность
seed = random.randint(1, 100000000)
print('seed =', seed)
# random.seed(19680801)

# фиксируем время начала работы программы
start_time = clock()

# число агентов
n_agents = 10
# размерность пространства
n_axis = 2
# порядок системы
n_dim = 2

# начальные условия в системе выбираем случайными
x0 = random.randn(n_agents * n_axis * n_dim)

# время начала моделирования
t_start = 0
# время конца моделирования
t_end = 5
# шаг по времени
t_delta = 0.01
# лист значений времени
t = arange(t_start, t_end, t_delta)

# радиус видимости агентов
vision_radius = 1.5

# интегрирование
x = odeint(model_second_order, x0, t, args=(vision_radius, n_agents, n_axis), tfirst=True)

# печатаем время выполнения основной части программы
print('time =', clock() - start_time)

# представляем решение в удобном виде
x = x.reshape(t.size, n_dim, n_axis, n_agents)

# рисуем графики
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
        plt.plot(t[:], x[:, i_dim, i_axis, :])
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
y = (sum(x[:, 1, :, :], axis=(2,)) / n_agents)
plt.plot(t, y)
plt.show()

y = y.reshape(t.size, n_axis, 1)
q = einsum('...k, ...k', x[:, 1, :, :] - y, x[:, 1, :, :] - y)
plt.plot(t, q)
plt.show()

q = einsum('...k, ...k', x[:, 1, :, 0:-2] - x[:, 1, :, 1:-1], x[:, 1, :, 0:-2] - x[:, 1, :, 1:-1])
plt.plot(t, q)
plt.show()
