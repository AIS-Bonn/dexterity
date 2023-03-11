import torch


# computes hyperbolic reward functions: r = scale * (1 / (dist + c))**pow
@torch.jit.script
def hyperbole_rew(scale, dist, c, pow=1):
    # type: (float, Tensor, float, int) -> Tensor
    rew = scale * (1 / (dist + c)).pow(pow)
    return rew


# computes squared hyperbolic reward functions: r = scale * (1 / (dist^2 + c))
@torch.jit.script
def sq_hyperbole_rew(scale, dist, c=1.0):
    # type: (float, Tensor, float) -> Tensor
    rew = scale * (1 / (dist.pow(2) + c))
    return rew


# computes exponential reward functions: r = scale * exp(-c * dist)
@torch.jit.script
def exponential_rew(scale, dist, c):
    # type: (float, Tensor, float) -> Tensor
    rew = scale * torch.exp(-c * dist)
    return rew

