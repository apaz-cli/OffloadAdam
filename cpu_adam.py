from typing import Any, Callable, Iterable, Optional

import torch
import torch.optim.optimizer
from torch.optim import Adam
from torch.optim.optimizer import ParamsT

import offload_adam


class CPUAdam(torch.optim.Optimizer):
    """
    Is not a torch.optim.Optimizer because of param groups.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps=1e-8,
    ):
        super().__init__(params, defaults=dict(lr=lr, betas=betas, eps=eps))
        self.state["optimizers"] = {}

        for param in params:
            self.state["optimizers"][param] = offload_adam.create_optimizer(param.grad, lr, betas[0], betas[1], eps)

    # TODO: Implement
    def __setstate__(self, state: dict[str, Any]):
        pass

    # TODO: Implement
    def __getstate__(self) -> dict[str, Any]:
        pass

    def step(self, closure: Optional[Callable[[], float]] = None):

        # I'm honestly really not sure why this is part of the API.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Step each parameter
        for group in self.param_groups: 
            for p in group['params']:
                self.step_param(p)

        return loss
                
    def step_param(self, param: torch.Tensor) -> None:
        param_opt = self.state["optimizers"].get(id(param))
        if type(param_opt) is not offload_adam.AdamOptimizer:
            raise ValueError("Parameter not registered with this optimizer")
        
    # NOTE: Not implementing zero_grad, should be handled by superclass, and
    #       doesn't require modifying optimizer state.

    def __del__(self):
        """Free the memory held by C++. Otherwise we risk leaking unholy amounts of memory."""
        for opt in self.state["optimizers"].values():
            offload_adam.destroy_optimizer(opt)
        self.state["optimizers"].clear()

    @classmethod
    def vector_width(cls) -> int:
        """
        Returns 1 if using the naive scalar implementation, 256 for avx2, 512 for avx512.
        """
        return offload_adam.vector_width()
