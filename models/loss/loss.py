import torch
from statistics import mean
from functools import partial
import torch.backends.cudnn as cudnn

def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x

def global_cosine_hm_adaptive(a, b, y=3):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()
        mean_dist = point_dist.mean()
        factor = (point_dist/mean_dist)**(y)
        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        partial_func = partial(modify_grad_v2, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss