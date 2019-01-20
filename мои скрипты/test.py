from numpy import *
from scipy.spatial import distance
from time import clock
import networkx as nx

# число агентов
n_agents = 4
# размерность пространства
n_axis = 2
# порядок системы
n_dim = 2

radius = 1.5
start_time = clock()
step_time = clock()
seed = random.randint(1, 1000000000)
random.seed(seed)
cnt = 0
cnt_dis = 0


while 1:
    if clock() > step_time + 10:
        step_time = clock()
        n_agents += 1
    system = random.randn(n_agents * n_axis * n_dim).reshape(n_dim, n_axis, n_agents)

    distance_matrix = distance.squareform(distance.pdist(system[0, :, :].transpose()))

    adjacency_matrix = int_(distance_matrix <= radius)

    graph = nx.from_numpy_array(adjacency_matrix)

    if not nx.is_connected(graph):
        cnt_dis += 1
        continue
    cnt += 1
    a22 = adjacency_matrix / sum(adjacency_matrix, axis=1).reshape(n_agents, 1) - eye(n_agents)

    zet = eye(n_agents, n_agents) - ones((n_agents, n_agents)) / n_agents

    q = dot(zet, a22)

    v = q.transpose() + q
    """
    print(a22)
    print(zet)
    print(q)
    print(dot(q, ones((n_agents, 1))))
    print(v)
    """
    r, rv = linalg.eig(v)
    # linalg.eig(v)
    if any(r > 0.000001):
        print(r)
        print(rv)
        print(seed)
        print(a22)
        print(q)
        print(v)
        break
    if clock() > start_time + 500:
        break
print(cnt, cnt_dis)

# добавить моделирование при плохой ситуации
