import numpy as np 
import sys
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import random

np.set_printoptions(threshold=sys.maxsize)
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

nx = 60
ny = 60
# x = np.linspace(0, 1, nx)
# y = np.linspace(0, 1, ny)
# xv, yv = np.meshgrid(x, y, indexing='ij')
shape = (nx,ny)
scale = 6

def cal_occ(map):
    occ = np.zeros(shape)
    for i in range(nx):
        for j in range(ny):
            n_occupied=0
            for delta in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]:
                if (i+delta[0] >=0 and i+delta[0] < nx and j+delta[1] >=0 and j+delta[1] <ny):
                    n_occupied += map[i+delta[0], j+delta[1]]
                else:
                    n_occupied +=0
            occ[i, j] = n_occupied
    return occ

def cal_entropy(filled_occ):
    H = -1*filled_occ*np.log2(filled_occ) + -1*(1-filled_occ)*np.log2(1-filled_occ)
    mask1 = map==1
    H[mask1] = 0
    H[np.isnan(H)] = 0
    return (H)


def plot_map(map):
    fig, ax = plt.subplots()
    # define the colors
    plt.imshow(map, cmap=plt.cm.gray_r)
    plt.show()


def scale_down(H_map):
    new_size = int(nx/scale)
    scaled_map = np.zeros([new_size ,new_size ])
    for i in range(new_size):
        for j in range(new_size):
            tmp = H_map[i*scale:i*scale+scale, j*scale:j*scale+scale]
            scaled_map[i,j] = tmp.sum()
    # plot_map(scaled_map)
    fig = px.imshow(scaled_map)
    fig.show()

def check_inbound(pos, step):
    x_condition = pos[0]+step[0] >=0 and pos[0]+step[0] < 5 
    y_condition = pos[1]+step[1] >=0 and pos[1]+step[1] < 5
    return x_condition and y_condition


def plot_complexity_change(record):
    # data to be plotted
    x = np.arange(0, len(record))
    y = np.array(record)
    plt.title("Line graph")
    plt.xlabel("Step")
    plt.ylabel("Complexity")
    plt.plot(x, y, color ="red")
    plt.show()
 
def build_entropy_map(map):
    pos = (0,0)
    explored = np.zeros(map.shape)
    complexity_list = []
    dir = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
    while pos[1]<map.shape[1]:
        # print(pos)
        explored[(pos)] = 1
        for delta in dir:
            if check_inbound(pos, delta):
                explored[tuple(np.add(pos, delta))] = 1
        
        expanding_map = explored*map
        H = cal_entropy(cal_occ(expanding_map)/8)
        # print(H.sum()/explored.sum())
        complexity_list.append(H.sum()/explored.sum())

        if pos[0] < map.shape[0]-1:
            pos = tuple(np.add(pos, (1,0)))
        else:
            pos = tuple(np.add(pos, (0,1)))
            pos = list(pos)
            pos[0] = 0
            pos = tuple(pos)
    plot_complexity_change(complexity_list)
    # print(explored)


map = np.random.choice([0,1], size=shape, p=[0.9, 0.1])
# map = pd.read_excel('map.xlsx').to_numpy()
H = cal_entropy(cal_occ(map)/8)
print(H)
print(H.sum()) 
print(H.sum()/3600) 

# scale_down(H)

build_entropy_map(map)
plot_map(map)