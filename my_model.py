from numpy import *
from scipy.spatial import distance


def model_second_order_with_delay(t, in_flat_system, radius, n_agents, n_dims=2, delay=0):
    """
        Оптимизированная версия с задержкой
        На вход принимает систему вида x = [x1, x2, ..., dot_x1, dot_x2, ...]
        x1, x2, ... - координаты агентов
        dot_x1, dot_x2, ... - скорости агентов
        n_agents - число агентов (больше 0)
        n_order - порядок матмодели агента (всегда 2)
        n_dims - размерность пространства (больше 0)
        возвращает dot_x = f(x)
    """
    # n_order - порядок матмодели агента (всегда 2)
    n_order = 2

    # система в удобном виде
    if delay > 0:
        flat_system = in_flat_system(t - delay)
    elif delay == 0:
        flat_system = in_flat_system
    system = flat_system.reshape(n_order, n_dims, n_agents)

    # считаем расстояние между агентами
    distance_matrix = distance.squareform(distance.pdist(system[0, :, :].transpose()))

    # граф видимости агентов друг другу
    graph = int_(distance_matrix <= radius)

    # матрица изменения скорости агентов
    # стремимся достичь средней скорости
    # однако доступны для вычисления управления лишь видимые агенты
    a22 = graph / sum(graph, axis=1).reshape(n_agents, 1) - eye(n_agents)

    # матрица изменения скорости всей системы
    a = block([[zeros((n_agents * n_dims, n_agents * n_dims)), eye(n_agents * n_dims)],
               [zeros((n_agents * n_dims, n_agents * n_dims)), kron(eye(n_dims), a22)]])

    # возвращаем f(x) = a * x
    return dot(a, flat_system)


def model_second_order_with_delay_and_fixed_graph(t, in_flat_system, radius, n_agents, n_dims=2, delay=0):
    """
        Оптимизированная версия с задержкой
        На вход принимает систему вида x = [x1, x2, ..., dot_x1, dot_x2, ...]
        x1, x2, ... - координаты агентов
        dot_x1, dot_x2, ... - скорости агентов
        n_agents - число агентов (больше 0)
        n_order - порядок матмодели агента (всегда 2)
        n_dims - размерность пространства (больше 0)
        возвращает dot_x = f(x)
    """
    # n_order - порядок матмодели агента (всегда 2)

    # система в удобном виде
    if delay > 0:
        flat_system = in_flat_system(t - delay)
    elif delay == 0:
        flat_system = in_flat_system

    # граф видимости агентов друг другу
    graph = array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                   [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 1, 1]])

    # матрица изменения скорости агентов
    # стремимся достичь средней скорости
    # однако доступны для вычисления управления лишь видимые агенты
    a22 = graph / sum(graph, axis=1).reshape(n_agents, 1) - eye(n_agents)

    # матрица изменения скорости всей системы
    a = block([[zeros((n_agents * n_dims, n_agents * n_dims)), eye(n_agents * n_dims)],
               [zeros((n_agents * n_dims, n_agents * n_dims)), kron(eye(n_dims), a22)]])

    # возвращаем f(x) = a * x
    return dot(a, flat_system)

