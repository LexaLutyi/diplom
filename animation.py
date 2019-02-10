from scipy.integrate import ode
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import clock

from my_model import model_second_order_with_delay_and_fixed_graph


# фиксируем случайность
random.seed(19680801)


def update(frame_number):
    tau = (frame_number + 1) * t_delta
    y = r.integrate(tau)
    yy = y.reshape(n_dim, n_axis, n_agents)
    scat.set_offsets(stack((yy[0, 0, :], yy[0, 1, :]), axis=1))
    print(tau, round(clock() - start_time, 2))


# число агентов
n_agents = 10
# размерность пространства
n_axis = 2
# порядок системы
n_dim = 2

# начальные условия в системе выбираем случайными
y0 = random.randn(n_agents * n_axis * n_dim)

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

# инициализируем интегратор
r = ode(model_second_order_with_delay_and_fixed_graph)
r.set_initial_value(y0, t_start).set_f_params(vision_radius, n_agents, n_axis)

# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(-10, 10), ax.set_xticks([])
ax.set_ylim(-10, 10), ax.set_yticks([])

# представляем начальные условия в удобном виде
yy0 = y0.reshape(n_dim, n_axis, n_agents)

# рисуем стартовые кружочки
scat = ax.scatter(yy0[0, 0, :],             # координаты агентов по x
                  yy0[0, 1, :],             # координаты агентов по y
                  s=1*vision_radius*1000,   # радиус кружка соответствует радусу видимости агента
                  lw=0.5,                   # толщина контура
                  edgecolors='black',       # цвет контура
                  facecolors='none'         # цвет заливки
                  )

start_time = clock()
# инициализация анимации
animation = FuncAnimation(fig, update, interval=10)

# словарь названий осей
label_dict = {
    (0, 0): 'x',
    (1, 0): 'x\'',
    (0, 1): 'y',
    (1, 1): 'y\'',
    (0, 2): 'z',
    (1, 2): 'z\''
}

# запуск анимации
plt.show()


