import torch
import torch.nn as nn

__all__ = ['device', 'FLOAT', 'LONG', 'DOUBLE', 'to_device', 'init_module', 'get_flat_grad_params',
           'resolve_activate_function']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.FloatTensor')

FLOAT = torch.FloatTensor
LONG = torch.LongTensor
DOUBLE = torch.DoubleTensor

def to_device(*params):
    return [x.to(device) for x in params]

def resolve_activate_function(name):
    if name.lower() == "relu":
        return nn.ReLU
    if name.lower() == "sigmoid":
        return nn.Sigmoid
    if name.lower() == "leakyrelu":
        return nn.LeakyReLU
    if name.lower() == "prelu":
        return nn.PReLU
    if name.lower() == "softmax":
        return nn.Softmax
    if name.lower() == "tanh":
        return nn.Tanh