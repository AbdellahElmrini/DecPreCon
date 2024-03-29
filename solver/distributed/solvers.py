import logging

from solver.distributed.extra import Extra, Extra_Cheb
from solver.distributed.nids import Nids
from solver.distributed.gd import GD
from solver.distributed.dpc import Dpc, Dpc_regression

class AvailableSolvers(object):
    solvers  = {
                "EXTRA": Extra,
                "EXTRACheb": Extra_Cheb,
                "NIDS": Nids,
                "DPC": Dpc,
                "GD": GD,
                "DPC-R":Dpc_regression
                }

    log = logging.getLogger("Args")

    @staticmethod
    def get(solver_name):
        if type(solver_name) is list:
            return [s for s in 
            AvailableSolvers.get(solver_name) if s is not None]
        
        if type(solver_name) is dict:
            return AvailableSolvers.get(list(solver_name.keys()))

        if type(solver_name) is str:
            solver_type = solver_name.split("_")[0]
            solver = AvailableSolvers.solvers.get(solver_type)
            if solver is None:
                AvailableSolvers.log.warning(f"Could not find solver {solver_type}")
                return [SolverNotFound, SolverNotFound]
            return solver


class SolverNotFound(object):
    def __init__(self, **kwargs):
        pass 

    def solve(self, nb_epochs):
        pass