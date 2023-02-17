import numpy as np

def cal_occ(map):
    occ = np.zeros((5, 5))
    for i in range(len(map)):
        for j in range(len(map[i])):
            n_occupied=0
            for d_i, d_j in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]:
                if i+d_i >=0 and i+d_i <=4 and j+d_j >=0 and j+d_j <=4:
                    n_occupied += map[i+d_i, j+d_j]
                else:
                    n_occupied +=0
            occ[i, j] = n_occupied
    print(occ[2,2])
    return occ

def cal_entropy(filled_occ):
    H = -1*filled_occ*np.log2(filled_occ) + -1*(1-filled_occ)*np.log2(1-filled_occ)
    mask1 = x==1
    H[mask1] = 0
    H[np.isnan(H)] = 0
    return (H)


x = np.zeros((5, 5))
x[3,1] =1
x[0,3] =1
x[2,2] =1
x[3,4] =1
x[0,1] =1
x[1,0] =1
x[4,3] =1
x[2,0] =1
x[3,3] =1
x[2,4] =1
x[0,0] =1
x[4,1] =1

H = cal_entropy(cal_occ(x)/8)
print(H)
print(H.sum()) 