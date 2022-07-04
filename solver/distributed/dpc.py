from mpi4py import MPI
import numpy as np



from solver.distributed.distributed_solver import DistributedSolver
 

class Dpc(DistributedSolver):
    def __init__(self, batch_factor=1., delta = 0.1,  **kwargs):
        super(Dpc, self).__init__(**kwargs)
        self.log.info(f"batch_factor: {batch_factor}")
        self.delta = delta
        self.nb_nodes = self.comm.size
        self.y = np.zeros(self.x.shape)
        self.omega = np.zeros(self.x.shape)
        self.step_size = 1. / (self.model.c + batch_factor * np.sum(self.model.get_smoothnesses(self.dataset))) # TODO calculate the real step size
        self.g = 0 # TODO use the dataset to create g
        self.gr_g = 0 # TODO use the dataset to create gr_g the gradient of g
        self.F = self.model.get_gradient(self.x, self.dataset) # TODO 

    def communication_step(self, omega):
        return self.multiply_by_w(omega) / self.graph.max_eig
        
    def run_step(self):
        eta = self.step_size
        psi = lambda z: 1/(2*eta) *np.linalg.norm(z-self.x)**2 + eta/2 * np.linalg.norm(z)**2 + \
                1/eta *(self.g - self.g - self.gr_g*(z-self.x)) - \
                z@(self.F + 1/eta *self.gr_g - (1-eta/self.c)*self.y +eta/self.c *self.omega)
        self.x -= 0 # TODO implement the iteration self.x -= argminimum(psi)
        self.y -= eta*(self.x + (self.y + self.omega)/self.model.c)
        self.omega -= eta*self.communication_step(self.omega)
        self.g = 0 # TODO update g
        self.gr_g = 0 # TODO update gr_g
        self.F = self.model.get_gradient(self.x, self.dataset) # TODO 