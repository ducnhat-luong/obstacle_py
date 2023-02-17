import numpy as np
import random

pos = (0,0)
explored = np.zeros((5,5))

def check_inbound(pos, step):
    x_condition = pos[0]+step[0] >=0 and pos[0]+step[0] < 5 
    y_condition = pos[1]+step[1] >=0 and pos[1]+step[1] < 5
    return x_condition and y_condition

dir = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]
for i in range(15):
    print(pos)
    explored[(pos)] = 1
    for delta in dir:
        if check_inbound(pos, delta):
            explored[tuple(np.add(pos, delta))] = 1
    
    next_step = random.choice(dir)
    while not check_inbound(pos, next_step):
        next_step = random.choice(dir)
    pos = tuple(np.add(pos, next_step))
print(explored)