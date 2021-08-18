from random import randint, random
import numpy as np


n = 5
m = 2
L = np.array([8., 6., 7., 4., 5.])
S = np.sum(L)/m

def F(subsets_old):
    subsets = list(subsets_old)
    m = len(subsets)
    random_positions = [];
    for i in range(m):
        random_positions.append([randint(-1, len(subsets[i])-1), randint(0, m-1)])
    
    for i in range(len(random_positions)):
        
        elem_take = random_positions[i][0]
        list_take = random_positions[i][1]
        
        if i != list_take and elem_take != -1 and len(subsets[list_take])-1 >= elem_take and len(subsets[list_take])>1:
            subsets[i] = subsets[i] + [subsets[list_take][elem_take]]
            l1 = subsets[list_take]
            if len(l1) > elem_take:
                l1 = l1[:elem_take] + l1[elem_take+1:]
            else:
                l1 = l1[:elem_take]
            subsets[list_take] = l1
            if type(subsets[list_take]) is np.float64:
                subsets[list_take] = [subsets[list_take]]
    return subsets

def cost_function(subsets):
    m = len(subsets)
    S = np.sum(np.sum(subsets))/m
    return sum((np.sum(subsets[i]) - S)**2 for i in range(m))

def T_Bolzman(T0, k):
    return T0/np.log(1 + k)

def T_Koshi(T0, k):
    return T0/k

def simulated_annealing(s1, tmin, tmax, T_func=T_Bolzman):
    history = [s1,]
    ti = tmax
    best_s = s1[:]
    si = s1[:]
    i = 1
    while ti > tmin:
        sc = F(si)
        delta_E = cost_function(sc) - cost_function(si)
        if delta_E < 0:
            si = sc.copy()
            history.append(si)
            if cost_function(si) < cost_function(best_s):
                best_s = si.copy()
        else:
            P = np.exp(-delta_E/ti)
            if P >= random():
                si = sc.copy()
                history.append(si)
                if cost_function(si) < cost_function(best_s):
                    best_s = si.copy()
        ti = T_func(ti, i)
        i += 1
    return best_s, history
