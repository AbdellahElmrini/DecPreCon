from numba import njit

@njit
def sparse_dot(x, y_ind, y_data):
    s = 0. 
    for i, y_i in zip(y_ind, y_data):
        s += x[i] * y_i
    return s 

@njit
def sparse_add(x, sy, coeff):
    for i, y_i in sy:
        x[i] += y_i