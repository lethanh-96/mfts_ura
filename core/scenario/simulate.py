import simulator
import torch

def simulate(args):
    with torch.set_grad_enabled(False):
        solver = simulator.create_solver(args)
        solver.simulate()
