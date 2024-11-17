from .weighted_random import WeightedRandomSolver
from .mftts0 import MeanFieldTransferThompsonSampling0Solver
from .mftts import MeanFieldTransferThompsonSamplingSolver
from .tts import TransferThompsonSamplingSolver
from .tcts import TransferCThompsonSamplingSolver
from .tmts import TransferMThompsonSamplingSolver
from .centralized import CentralizedSolver
from .ts import ThompsonSamplingSolver
from .mabmfg import MabMfgSolver
from .greedy import GreedySolver
from .hedge import HedgeSolver
from .mfdqn import MfdqnSolver
from .base import BaseSolver

def create_solver(args):
    if args.solver == 'random':
        return BaseSolver(args)
    elif args.solver == 'ts':
        return ThompsonSamplingSolver(args)
    elif args.solver == 'tts':
        return TransferThompsonSamplingSolver(args)
    elif args.solver == 'tcts':
        return TransferCThompsonSamplingSolver(args)
    elif args.solver == 'tmts':
        return TransferMThompsonSamplingSolver(args)
    elif args.solver == 'mftts':
        return MeanFieldTransferThompsonSamplingSolver(args)
    elif args.solver == 'mftts0':
        return MeanFieldTransferThompsonSampling0Solver(args)
    elif args.solver == 'hedge':
        return HedgeSolver(args)
    elif args.solver == 'centralized':
        return CentralizedSolver(args)
    elif args.solver == 'mabmfg':
        return MabMfgSolver(args)
    elif args.solver == 'greedy':
        return GreedySolver(args)
    elif args.solver == 'weighted_random':
        return WeightedRandomSolver(args)
    elif args.solver == 'mfdqn':
        return MfdqnSolver(args)
    else:
        raise NotImplementedError
