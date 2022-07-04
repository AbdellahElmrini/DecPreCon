import numpy as np
from numba import njit

class LinearRegression(Object):

    def __init__(self) -> None:
        super().__init__()

    def get_L(self, dataset):
        pass

    def get_smoothness(self, dataset):
        pass

    def get_gradient(self, dataset):
        pass

    def compute_error(self, theta, dataset):
        return 0

    def get_global(self, comm_size):
        return LinearRegression()
        