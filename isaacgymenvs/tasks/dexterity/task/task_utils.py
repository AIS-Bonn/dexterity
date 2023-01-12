import torch


# compute hyperbolic reward functions of the type: r = scale * (1 / (dist + c))**pow
@torch.jit.script
def hyperbole_rew(scale, dist, c, pow=1):
    # type: (float, Tensor, float, int) -> Tensor
    rew = scale * (1 / (dist + c)).pow(pow)
    return rew

