from typing import Iterable
import torch
import offload_adam

class CPUAdam:
    def __init__(self, opt):
        self.optimizer = opt
        self.param = None
        self.grad = None
    
    def step(self, param: torch.Tensor, grad: torch.Tensor):
        """Execute one optimization step"""
        self.param = param
        self.grad = grad
        return offload_adam.step(self.optimizer, param, grad)
    
    def zero_grad(self, set_to_none = True):
        """Zero out the gradient"""
        if self.grad is not None:
            if set_to_none:
                self.grad = None
            else:
                self.grad.zero_()
    
    def __del__(self):
        """Clean up the C++ optimizer"""
        if hasattr(self, 'optimizer'):
            offload_adam.destroy_optimizer(self.optimizer)


def construct_for_parameters(params: Iterable[torch.Tensor] | torch.Tensor, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False) -> list[CPUAdam]:
    beta1, beta2 = betas

    opts: list[CPUAdam] = []
    params = (params,) if isinstance(params, torch.Tensor) else params
    for p in params:
        assert p.grad is not None
        assert p.grad.device.type == 'cpu'

        off_opt = offload_adam.create_optimizer(p.grad, lr, beta1, beta2, eps, weight_decay, amsgrad)
        opt = CPUAdam(off_opt)
        opts.append(opt)

    return opts
