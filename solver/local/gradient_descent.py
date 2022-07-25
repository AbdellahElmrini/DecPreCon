import numpy as np

from solver.solver import Solver


class GD(Solver):
    def __init__(self, **kwargs):
        super(GD, self).__init__(**kwargs)
        self.L = self.model.get_L(self.dataset)
        self.set_regularization(self.model.c)

    def set_regularization(self, lam):
        self.step_size = 1.0 / (self.L + lam) 
        self.lam = lam

    def run_step(self, grad, nb_iters):
        # print(np.linalg.norm(self.model.get_gradient_without_reg(self.x, self.dataset) + grad + self.lam * self.x))
        for _ in range(10 * nb_iters):
            self.x = (1 - self.step_size * self.lam) * self.x - self.step_size * (self.model.get_gradient_without_reg(self.x, self.dataset) + grad)
        self.subproblem_grad_norm = np.linalg.norm(self.model.get_gradient_without_reg(self.x, self.dataset) + grad + self.lam * self.x)
        # print(self.subproblem_grad_norm)
        return np.copy(self.x)

class AGD(Solver):   
    def __init__(self, **kwargs):
        super(AGD, self).__init__(**kwargs)
        self.lam = self.model.c
        self.L = self.model.get_L(self.dataset)
        self.set_regularization(self.model.c)
        self.y = np.copy(self.x)
        self.subproblem_grad_norm = 0.

    def set_regularization(self, lam):
        self.step_size = 1. / (self.L + lam) 
        sqrt_kappa = np.sqrt(1. / (self.step_size * lam))
        self.beta = (sqrt_kappa - 1) / (sqrt_kappa + 1)
        self.lam = lam

    def run_step(self, grad, nb_epochs):
        self.subproblem_grad_norm = np.inf
        inner_repeats_count = 0
        while inner_repeats_count < self.max_inner_repeats:
            if self.subproblem_grad_norm < self.inner_precision:
                break 
            for _ in range(nb_epochs):
                subproblem_grad = self.model.get_gradient_without_reg(self.x, self.dataset) + grad + self.lam * self.x
                y_new = self.x - self.step_size * subproblem_grad 
                self.x = (1 + self.beta) * y_new - self.beta * self.y 
                self.y = y_new

            self.subproblem_grad_norm = np.linalg.norm(subproblem_grad)
            inner_repeats_count += 1

        # Safer to copy (although it is more expensive)
        return np.copy(self.x)