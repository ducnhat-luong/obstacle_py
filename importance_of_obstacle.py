import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn.preprocessing import normalize

NUM_X = 50
NUM_Y = 25
np.random.seed(seed=5)
P_OBS = 0.04


pos=(0.25, 0.25)
xs_src = np.linspace(0, 2, NUM_X)
ys_src = np.linspace(0, 1, NUM_Y)

resolution=0.000001

angle_map = np.zeros((NUM_X, NUM_Y))
result_map = np.zeros((NUM_X, NUM_Y))

# dx = np.diff(xs_src).mean()
# dy = np.diff(ys_src).mean()

# extent_x = [xs_src[0] - dx/2, xs_src[-1] + dx/2]
# extent_y = [ys_src[0] - dy/2, ys_src[-1] + dy/2]
# extent = extent_x + extent_y


# convert xs_src & ys_src
xs_src_, ys_src_ = np.meshgrid(xs_src, ys_src, indexing='ij')
# dx = pos[0] - xs_src_
# dy = pos[1] - ys_src_

# # round dx's and dy's less than resolution down to zero
# dx[np.abs(dx) < resolution] = 0
# dy[np.abs(dy) < resolution] = 0

# print(dx)
# x_axis = [1,0]
# vector_1 = np.array([[1, 2], [4, 5]], np.int32)
# unit_vector_1 = vector_1 / np.linalg.norm(vector_1)


def get_obs():
    obs = np.random.rand(NUM_X,NUM_Y)
    obs =  obs < P_OBS
    obs_angle = np.zeros((NUM_X, NUM_Y))
    obs_angle[~obs] = 1
    ax = sns.heatmap(obs_angle.T)
    ax.invert_yaxis()
    plt.title("Random Obstacles")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return obs


def get_obs_ang_1(obs):
    for i in range(NUM_X):
        for j in range(NUM_Y):
            src = [xs_src_[i,j], ys_src_[i,j]]
            impact_list = []
            for u in range(NUM_X):
                for v in range(NUM_Y):
                    if obs[u,v]:
                        src = [xs_src_[i,j], ys_src_[i,j]]
                        obs_cell = [xs_src_[u,v], ys_src_[u,v]]
                        angle = get_angle(src, obs_cell, pos)
                        if angle > np.pi/2: 
                            impact = 0 
                        else:
                            impact = np.pi/2 - angle
                        impact_list.append(impact)
            if len(impact_list) > 0:
                result_map[i,j] = sum(impact_list) 


def get_angle(src, obs, agent):
    vector_1 = np.array(agent) - np.array(src)
    vector_2 = np.array(obs) - np.array(src)
    
    if np.all(vector_2 == 0):
        return 0

    vector_1 = list(vector_1)
    vector_2 = list(vector_2)
    len_as = np.linalg.norm(vector_1)
    len_os = np.linalg.norm(vector_2)

    if len_os > len_as:
        return np.pi

    unit_vector_1 = vector_1 / np.linalg.norm(len_as)
    unit_vector_2 = vector_2 / np.linalg.norm(len_os)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    cos = round(dot_product, 4)
    angle = np.arccos(cos)
    return angle

# def normalize(matrix):
#     row_sums = matrix.sum(axis=1)
#     new_matrix = matrix / row_sums[:, np.newaxis]
#     return new_matrix


fig, ax = plt.subplots()
# get_angle_map()
obs = get_obs()
get_obs_ang_1(obs)
result_map = result_map/ np.amax(result_map)
ax = sns.heatmap(result_map.T)
ax.invert_yaxis()
plt.title('The clean source map')
plt.xlabel('x')
plt.ylabel('y')
plt.show()