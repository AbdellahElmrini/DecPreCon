from mpi4py import MPI
import numpy as np
from scipy.sparse.linalg import inv



from solver.distributed.distributed_solver import DistributedSolver
from solver.local.sdca import SDCA
from solver.local.gradient_descent import GD, AGD
 

class Dpc(DistributedSolver):
    def __init__(self, batch_factor=1., inner_precision =1e-6, max_inner_repeats=20, inner_epochs=20, **kwargs):
        super(Dpc, self).__init__(**kwargs)
        self.log.info(f"batch_factor: {batch_factor}")
        self.nb_nodes = self.comm.size
        self.y = np.zeros(self.x.shape)
        self.omega = np.zeros(self.x.shape)
        self.step_size = self.model.c/2
        self.L_g = np.sum(self.model.get_smoothnesses(self.shared_dataset))
        self.L_r = 10. # TODO UPDATE to the real relative smoothness (or to a proxy)
        self.g_reg = 1.

        self.local_solver = SDCA(
            model=self.model, 
            dataset = self.shared_dataset,
            seed= self.seed,
            inner_precision = inner_precision,
            max_inner_repeats = max_inner_repeats ) #TODO fix the local solver parameters 
        
        # self.local_solver = GD(
        #             model=self.model, 
        #             dataset = self.shared_dataset,
        #             seed= self.seed,
        #             inner_precision = inner_precision,
        #             max_inner_repeats = max_inner_repeats ) #TODO fix the local solver parameters 

        self.inner_epochs = inner_epochs
        self.C = self.L_r*self.model.c/2 
        self.alpha = self.step_size*self.model.c/(2 * self.L_g * self.C ) # TODO 

        print(f"alpha: {self.alpha}, c: {self.model.c}, C: {self.C}, L_g: {self.L_g}")
        assert(self.step_size * self.alpha < self.model.c)
        
        # self.local_solver.set_regularization(self.model.c+ np.power(self.step_size, 2) / (self.C* (self.model.c-self.step_size * self.alpha)))
        added_h_reg = np.power(self.step_size, 2) / (self.C* (1 -self.step_size * self.alpha/ self.model.c))
        self.local_solver.set_regularization(self.g_reg + added_h_reg)

        print(f"inner epochs: {self.inner_epochs}, inner reg: {self.g_reg + added_h_reg}")

    def communication_step(self, omega):
        return self.multiply_by_w(omega) / self.graph.max_eig
        
    def run_step(self):
        self.gr_F = self.model.get_gradient_without_reg(self.x, self.dataset)
        eta = self.step_size 
        sigma = self.model.c
        grad_h = self.model.get_gradient_without_reg(self.x, self.shared_dataset) + self.g_reg * self.x
        a = (eta/self.C) * (self.gr_F - eta/(sigma-eta*self.alpha) * self.omega + (sigma - eta*(1+self.alpha))/(1-eta*self.alpha/sigma) * self.y ) - grad_h 
        self.x = self.local_solver.run_step(a, self.inner_epochs) #TODO 
        if self.comm.rank == 0 and self.iteration_number % 1000 == 0:
            print(self.local_solver.subproblem_grad_norm)
        y_temp = self.y
        self.y = 1/(1-eta*self.alpha/sigma) * ((sigma- eta*(1+self.alpha))/sigma * self.y - eta/sigma*self.omega+eta*self.x )
        self.omega -= eta/sigma*self.communication_step(self.omega+y_temp)
        
        
       
class Dpc_regression(DistributedSolver):
    def __init__(self, batch_factor=1., **kwargs):
        super(Dpc_regression, self).__init__(**kwargs)
        self.log.info(f"batch_factor: {batch_factor}")
        self.nb_nodes = self.comm.size
        self.y = np.zeros(self.x.shape)
        self.omega = np.zeros(self.x.shape)
        self.eta = 1/2
        self.step_size = self.eta
        self.mu_r = 0.1
        self.L_r = 3000. # TODO UPDATE to the real relative smoothness (or to a proxy)
        self.alpha = self.mu_r/(4*self.L_r)
        self.Q = self.model.get_quadratic(self.shared_dataset)
        self.Q_inv = inv(self.Q)
        
    
    def communication_step(self, omega):
        return self.multiply_by_w(omega) / self.graph.max_eig
        
    def run_step(self):
        self.gr_F = self.model.get_gradient(self.x, self.dataset)
        x_temp = (self.L_r/2 + self.eta*self.mu_r/2)*self.x -  self.eta*self.Q_inv@self.gr_F- self.eta/(1-self.eta*self.alpha) * ((1- self.eta*self.alpha - \
            self.eta)*self.Q_inv@self.y -self.eta*self.Q_inv@self.omega )
        self.x = 1/(self.L_r/2 - self.eta**2/(1-self.eta*self.alpha))*x_temp
        omega_temp = self.omega
        self.omega -= self.eta*self.communication_step(self.omega+self.y)
        self.y = 1/(1-self.eta*self.alpha) *(((1- self.eta*self.alpha - self.eta)*self.y -self.eta*omega_temp +self.eta*self.Q@self.x))
