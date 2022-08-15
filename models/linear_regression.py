import numpy as np
from numba import njit
from scipy.sparse.linalg import eigs

class LinearRegression(object):
    name ="Lreg"
    def __init__(self, c):
        self.c = c
    

    def get_L(self, dataset):
        @njit 
        def get_L(indptr, data):
            L = 0
            for i in range(len(indptr) - 1):
                L = max(L, np.sum(np.power(data[indptr[i]:indptr[i+1]], 2)))
            return L

        return 700*max(eigs((dataset.X.T @ dataset.X).toarray())[0]) #100*get_L(dataset.X.T.indptr, dataset.X.T.data) / dataset.N

    def get_smoothnesses(self, dataset):
        def get_L(indptr, data):

            res = np.zeros(len(indptr) -1 )
            for i in range(len(indptr) - 1):
                vec = np.array(data[indptr[i]:indptr[i+1]])
                vec= np.expand_dims(vec, 1)
                print(np.linalg.eig(vec @ vec.T)[0])
                res[i] = np.max(np.linalg.eig(vec @ vec.T)[0])

            return res

        
        return 700*get_L(dataset.X.T.indptr, dataset.X.T.data) / dataset.N


    def get_gradient(self, theta, dataset):
        s =  dataset.X @ ((dataset.X.T@theta) - dataset.y)
        return self.c * theta + s / dataset.N


    def compute_error(self, theta, dataset):
        s = 1/2 * np.sum((dataset.X.T@theta) - dataset.y)**2
        return 0.5*self.c*(theta@theta) + s/dataset.N

    def get_quadratic(self, dataset):
        return 1/dataset.N *dataset.X @ dataset.X.T

    def get_global(self, comm_size):
        return LinearRegression(self.c*comm_size)
        