from mpi4py import MPI
import numpy as np
from scipy.optimize import minimize



from solver.distributed.distributed_solver import DistributedSolver
from solver.local.sdca import SDCA
from solver.local.gradient_descent import GD, AGD
 

class Dpc(DistributedSolver):
    def __init__(self, batch_factor=1., inner_precision =1e-5, max_inner_repeats = 40, inner_epochs=50, **kwargs):
        super(Dpc, self).__init__(**kwargs)
        self.log.info(f"batch_factor: {batch_factor}")
        self.nb_nodes = self.comm.size
        self.y = np.zeros(self.x.shape)
        self.omega = np.zeros(self.x.shape)
        self.step_size = self.model.c/2
        self.L_g = np.sum(self.model.get_smoothnesses(self.shared_dataset))
        self.L_r = 10. # TODO UPDATE to the real relative smoothness (or to a proxy)
        self.local_solver = SDCA(
            model=self.model, 
            dataset = self.shared_dataset,
            seed= self.seed,
            inner_precision = inner_precision,
            max_inner_repeats = max_inner_repeats ) #TODO fix the local solver parameters 
        
        self.inner_epochs = inner_epochs
        self.C = self.L_r*self.model.c/2 
        self.alpha = self.step_size*self.model.c/(2 * self.L_g * self.C ) # TODO 

        print(f"alpha: {self.alpha}, c: {self.model.c}, C: {self.C}, L_g: {self.L_g}")
        assert(self.step_size * self.alpha < self.model.c)
        self.local_solver.set_regularization(self.model.c+ np.power(self.step_size, 2) / (self.C* (self.model.c-self.step_size * self.alpha)))

    def communication_step(self, omega):
        return self.multiply_by_w(omega) / self.graph.max_eig
        
    def run_step(self):
        self.gr_F = self.model.get_gradient(self.x, self.dataset)
        eta = self.step_size 
        sigma = self.model.c
        C = self.C
        a = (eta/C ) * (self.gr_F - eta/(sigma-eta*self.alpha) *self.omega + (sigma - eta*(1+self.alpha))/(1-eta*self.alpha/sigma) * self.y)
        self.x = self.local_solver.run_step(a,self.inner_epochs) #TODO 
        self.y = 1/(1-eta*self.alpha/sigma) * ((sigma- eta*(1+self.alpha))/sigma * self.y - eta/sigma*self.omega+eta*self.x )
        self.omega -= eta/sigma*self.communication_step(self.omega+self.y)
       