import simulator
import torch

def simulate(args):
    if args.mode == 'test':
        with torch.set_grad_enabled(False):
            solver = simulator.create_solver(args)
            solver.simulate()
    elif args.mode == 'train':
        solver = simulator.create_solver(args)
        solver.simulate()
    else:
        raise NotImplementedError
