from mpi4py import MPI
import numpy as np
from scipy.optimize import minimize



from solver.distributed.distributed_solver import DistributedSolver
from solver.local.sdca import SDCA
 

class Dpc(DistributedSolver):
    def __init__(self, batch_factor=1., inner_precision =1e-5, max_inner_repeats = 40, inner_epochs=50, **kwargs):
        super(Dpc, self).__init__(**kwargs)
        self.log.info(f"batch_factor: {batch_factor}")
        self.nb_nodes = self.comm.size
        self.y = np.zeros(self.x.shape)
        self.omega = np.zeros(self.x.shape)
        self.step_size = 0.001 #. / (self.model.c + batch_factor * np.sum(self.model.get_smoothnesses(self.dataset))) # TODO calculate the real step size
        self.L_g = np.sum(self.model.get_smoothnesses(self.shared_dataset))
        self.L_r = self.L_g # TODO UPDATE to the real relative smoothness (or to a proxy)
        self.g = self.model.compute_error(self.x, self.shared_dataset)  # TODO 
        self.gr_g = self.model.get_gradient(self.x, self.shared_dataset)  # TODO 
        self.F = self.model.get_gradient(self.x, self.dataset) # TODO 
        self.local_solver = SDCA(
            model=self.model, 
            dataset = self.shared_dataset,
            seed= self.seed,
            inner_precision = inner_precision,
            max_inner_repeats = max_inner_repeats ) #TODO fix the model parameters 
        self.inner_epochs = inner_epochs
        #self.local_solver.set_regularization(self.model.c)

    def communication_step(self, omega):
        return self.multiply_by_w(omega) / self.graph.max_eig
        
    def run_step(self):
        self.g = self.model.compute_error(self.x, self.shared_dataset)# TODO
        self.gr_g = self.model.get_gradient(self.x, self.shared_dataset)  # TODO 
        self.F = self.model.get_gradient(self.x, self.dataset) # TODO 
        eta = self.step_size
        C = self.L_r/2 
        sigma = self.model.c
        alpha = eta*self.model.c/(2 * self.L_g*C ) # TODO verify alpha's expression
        self.h = C*self.g 
        self.gr_h = C*self.gr_g
        #psi = lambda z: 1/eta*self.h +  eta/(2*(1-eta*alpha/self.model.c)) * np.linalg.norm(z)**2 + \
                #z@(self.F - eta/(self.model.c-eta*alpha) *self.omega - 1/eta * self.gr_h + 1/(1-eta*alpha/self.model.c) * self.y - (eta*(1+alpha)*self.x)/(self.model.c - eta*alpha) )
        
        a = self.F - eta/(sigma-eta*alpha) *self.omega + (sigma - eta*(1+alpha))/(1-eta*alpha/sigma) * self.y
        
        self.x = self.local_solver.run_step(a,self.inner_epochs) #TODO Write the proper inner iteration
        #self.x = minimize(psi, np.zeros(len(self.x)), options={"maxiter" : 30}).x # TODO  
        self.y -= eta*(self.x + (self.y + self.omega)/self.model.c)
        self.omega -= eta*self.communication_step(self.omega+self.y)
       