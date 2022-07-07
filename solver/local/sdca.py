from numba import njit
from numba.typed import List
import numpy as np
import scipy.special

from solver.utils import sparse_dot
from solver.solver import Solver
 

@njit
def add_alpha(alpha, step_size, g, X_ind, X_data, g_fixed):
    for j, sparse_idx in enumerate(X_ind):
        alpha[j] = (1 - step_size) * alpha[j] - step_size * (g * X_data[j] + g_fixed[sparse_idx])

@njit
def add_w(w, alpha, step_size, g, X_ind, X_data, g_fixed):
    for j, sparse_idx in enumerate(X_ind):
        w[sparse_idx] -= step_size * (g * X_data[j] + g_fixed[sparse_idx] + alpha[j])

@njit
def solve_inner_problem_sdca(indptr, indices, data, dataset_y, random_indices, w, alpha, primal_step_size, dual_step_size, g_fixed, get_gradient):
    for i in random_indices:
        g = get_gradient(sparse_dot(w, indices[indptr[i]:indptr[i+1]], data[indptr[i]:indptr[i+1]]), dataset_y[i])
        add_w(w, alpha[i], primal_step_size, g, indices[indptr[i]:indptr[i+1]], data[indptr[i]:indptr[i+1]], g_fixed)
        add_alpha(alpha[i], dual_step_size, g, indices[indptr[i]:indptr[i+1]], data[indptr[i]:indptr[i+1]], g_fixed)


class SDCA(Solver):
    def __init__(self, **kwargs):
        super(SDCA, self).__init__(**kwargs)
        self.proba_coord = self.dataset.precompute_probas()
        self.zero_mask = self.proba_coord == 0
        self.nz_mask = ~self.zero_mask
        #A sequence of points with each having the same sparsity pattern as X_i, so we don't store the indices
        self.alpha = SDCA.initialize_alpha(self.dataset.X.T.indptr)
        self.L = self.model.get_L(self.dataset)
        self.lam = self.model.c
        self.set_regularization(self.model.c)
        self.subproblem_grad_norm = 0.
        
    def set_regularization(self, lam):
        self.primal_step_size = min(0.25 / self.L, 0.25 / (lam * self.dataset.N))
        self.dual_step_size = self.primal_step_size * lam * self.dataset.N
        self.x = self.lam * self.x / lam #rescales x to respect the primal-dual relationship
        self.lam = lam

    @staticmethod
    @njit
    def initialize_alpha(indptr):
        alpha = List()
        for i in range(len(indptr)-1):
            alpha.append(np.zeros(indptr[i+1] - indptr[i])) 
        return alpha

    def run_step(self, grad, nb_epochs):
        nb_iters = int(nb_epochs * self.dataset.N)
        self.subproblem_grad_norm = np.inf
        inner_repeats_count = 0
        while inner_repeats_count < self.max_inner_repeats:
            if self.subproblem_grad_norm < self.inner_precision:
                break
            solve_inner_problem_sdca(self.dataset.X.T.indptr, self.dataset.X.T.indices, self.dataset.X.T.data,
                                    self.dataset.y, 
                                    self.rs.randint(self.dataset.N, size=(nb_iters,)),
                                    self.x, self.alpha, self.primal_step_size, 
                                    self.dual_step_size, np.divide(grad, self.proba_coord, where=self.nz_mask),
                                    self.model.get_1d_gradient
                                    )
            self.x[self.zero_mask] = - grad[self.zero_mask] / self.lam
            self.subproblem_grad_norm = np.linalg.norm(self.model.get_gradient(self.x, self.dataset) + grad + (self.lam - self.model.c) * self.x)
            inner_repeats_count += 1
        # Safer to copy (although it is more expensive)
        return np.copy(self.x)