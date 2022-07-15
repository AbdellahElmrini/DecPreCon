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
        self.step_size = 0.01 #. / (self.model.c + batch_factor * np.sum(self.model.get_smoothnesses(self.dataset))) # TODO calculate the real step size
        self.L_g = np.sum(self.model.get_smoothnesses(self.shared_dataset))
        # Here it's unclear which relative smoothness you assume. Is it before rescaling h? After rescaling h?
        # This changes a lot of things. If it's after rescaling then it should be related to the step-size
        # but I think it's before. 
        self.L_r = 100. #self.L_g # TODO UPDATE to the real relative smoothness (or to a proxy)
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

        C = self.L_r/2 
        self.alpha = self.step_size*self.model.c/(2 * self.L_g*C ) # TODO verify alpha's expression

        # Adds regularization to the local problem solved.
        # The square comes from the fact that we consider D_h and not D_h / eta

        # Should also try with some regularization. 
        assert(self.step_size * self.alpha < self.model.c)
        self.local_solver.set_regularization(self.model.c + np.power(self.step_size, 2) / ( C * (self.model.c-self.step_size * self.alpha)))

    def communication_step(self, omega):
        return self.multiply_by_w(omega) / self.graph.max_eig
        
    def run_step(self):
        self.g = self.model.compute_error(self.x, self.shared_dataset)# TODO
        self.gr_g = self.model.get_gradient(self.x, self.shared_dataset)  # TODO 
        self.F = self.model.get_gradient(self.x, self.dataset) # TODO 
        eta = self.step_size
        C = self.L_r/2 
        sigma = self.model.c

        # This has no effect 
        self.h = C*self.g 

        self.gr_h = C*self.gr_g
        #psi = lambda z: 1/eta*self.h +  eta/(2*(1-eta*alpha/self.model.c)) * np.linalg.norm(z)**2 + \
                #z@(self.F - eta/(self.model.c-eta*alpha) *self.omega - 1/eta * self.gr_h + 1/(1-eta*alpha/self.model.c) * self.y - (eta*(1+alpha)*self.x)/(self.model.c - eta*alpha) )
        


        # Is the inner solver scaled well? I think we need to add this eta in front because the inner solver only solves D_h
        # Similarly, if you scaled h in the analysis, then you should scale it here as well.
        # Should we divide or multiply by constant C?

        # Overall, the main problem is that the note is not 100% clear, and so a line-by-line debugging is not that direct.
        # You should aim for very clear pseudo-code in order to have very clear code.  
        
        # Here I assumed scaling h meant multiplying it by C, and so I divided a (and the regularization in set_regularization) by C 
        # to cancel the effect
        
        a = (eta / C) * (self.F - eta/(sigma-eta*self.alpha) *self.omega + (sigma - eta*(1+self.alpha))/(1-eta*self.alpha/sigma) * self.y)
        


        self.x = self.local_solver.run_step(a,self.inner_epochs) #TODO Write the proper inner iteration
        #self.x = minimize(psi, np.zeros(len(self.x)), options={"maxiter" : 30}).x # TODO  
        self.y -= eta*(self.x + (self.y + self.omega)/self.model.c)
        self.omega -= eta*self.communication_step(self.omega+self.y)
       