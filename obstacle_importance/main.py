import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn.preprocessing import normalize

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


def get_wall():
    obs = np.zeros((NUM_X, NUM_Y))
    obs[24:26, 5:20] = 1
    ax = sns.heatmap(obs.T, cmap=sns.cm.rocket_r)
    ax.invert_yaxis()
    plt.title("Wall Obstacle")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return obs


def get_wall2():
    obs = np.zeros((NUM_X, NUM_Y))
    obs[24:26, 1:5] = 1
    obs[24:26, 20:25] = 1
    ax = sns.heatmap(obs.T, cmap=sns.cm.rocket_r)
    ax.invert_yaxis()
    plt.title("Wall Obstacle")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return obs

def get_wall3():
    obs = np.zeros((NUM_X, NUM_Y))
    obs[0:1,:] = 1
    obs[49:50,:] = 1
    obs[:, 0:1] = 1
    obs[:, 24:25] = 1
    obs[24:26, 0:18] = 1
    ax = sns.heatmap(obs.T, cmap=sns.cm.rocket_r)
    ax.invert_yaxis()
    plt.title("Wall Obstacle")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return obs


# impact = clean
def get_impact_map(obs):
    for i in range(NUM_X):
        for j in range(NUM_Y):
            src = [xs_src_[i,j], ys_src_[i,j]]
            impact_list = []
            for u in range(NUM_X):
                for v in range(NUM_Y):
                    if obs[u,v]:
                        obs_pos = [xs_src_[u,v], ys_src_[u,v]]
                        inside = check_in_rectangle(obs_pos, pos, src )
                        if not inside:
                            impact = 1
                            impact_list.append(impact)
                            continue

                        # inside = check_in_ellipse(obs_pos, pos, src)
                        # if not inside:
                        #     impact = 1
                        #     impact_list.append(impact)
                        #     continue

                        vector_sa = np.array(pos) - np.array(src)
                        vector_so = np.array(obs_pos) - np.array(src)

                        len_sa = np.sqrt(vector_sa.dot(vector_sa))
                        len_so = np.sqrt(vector_so.dot(vector_so))

                        if np.all(vector_so == 0):  # check zero vector (source duplicate with obstacle)
                            impact = 0
                        elif len_so > len_sa:       # check if obstacle beyond the agent
                            impact = 1
                        else:  
                            angle_aso = get_angle(vector_sa, vector_so)
                            if angle_aso > np.pi/4: 
                                impact = 1 
                            else:
                                impact = angle_aso / (np.pi/4)
                        impact_list.append(impact)

            if len(impact_list) > 0:
                result_map[i,j] = sum(impact_list)/len(impact_list)
    

# param vector (numpy.ndarray): vector coordinate [x,y].
def get_unit_vector(vector):
    return vector / np.sqrt(vector.dot(vector))


# def check_in_ellipse(obs, agent, src):
#     vector_sa = np.array(pos) - np.array(src)
#     len_sa = np.sqrt(vector_sa.dot(vector_sa))
#     x_axis = np.array([1,0])
#     alpha = get_angle(vector_sa, x_axis)

#     center = (np.array(agent) + np.array(src))/2
#     center = list(center)
#     a = len_sa/2
#     b = 0.1

#     major_term = (obs[0]*np.cos(alpha) - obs[1]*np.sin(alpha) - center[0])**2
#     minor_term = (obs[0]*np.sin(alpha) + obs[1]*np.cos(alpha) - center[1])**2
#     p = major_term/(a**2) + minor_term/(b**2)

#     if p > 1:
#         return False
#     else: 
#         return True


def check_in_rectangle(obs, agent, src):
    vector_sa = np.array(agent) - np.array(src)
    vector_os = np.array(src) - np.array(obs)

    d = abs(np.cross(vector_sa, vector_os)/np.linalg.norm(vector_sa))
    if d < 0.2:
        return True
    else:
        return False


# param src (list): coordinate of source (x,y).
def get_angle(vec1,vec2):
    vec1 = vec1 / np.sqrt(vec1.dot(vec1))
    vec2 = vec2 / np.sqrt(vec2.dot(vec2))
    dot_product = np.dot(vec1, vec2)
    cos = round(dot_product, 4)
    angle = np.arccos(cos)
    return angle


if __name__ == "__main__":
    NUM_X = 50
    NUM_Y = 25
    P_OBS = 0.04

    pos=(1.7, 0.9)
    np.random.seed(seed=3)

    xs_src = np.linspace(0, 2, NUM_X)
    ys_src = np.linspace(0, 1, NUM_Y)
    xs_src_, ys_src_ = np.meshgrid(xs_src, ys_src, indexing='ij')
    resolution=0.000001

    angle_map = np.zeros((NUM_X, NUM_Y))
    result_map = np.zeros((NUM_X, NUM_Y))

    fig, ax = plt.subplots()
    # obs = get_obs()
    obs = get_wall3()
    get_impact_map(obs)
    result_map = result_map/ np.amax(result_map)
    result_map[obs==1] = 0
    result_map = result_map/ np.sum(result_map)
    ax = sns.heatmap(result_map.T)
    ax.invert_yaxis()
    plt.title('The clean source map')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()