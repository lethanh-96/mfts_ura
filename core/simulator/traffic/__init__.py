# from .ppp import PPP
from .sppp import SPPP
# from .iot_trace import IotTrace

def create_traffic_model(args):
    if args.traffic_model == 'sppp':
        return SPPP(args)
    else:
        raise NotImplementedError
