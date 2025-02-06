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
        for group in self.param_groups:
            for param in group["params"]:
                self.state[param] = offload_adam.create_optimizer(
                    param, lr, betas[0], betas[1], eps
                )

    # TODO: Implement
    def __setstate__(self, state: dict[str, Any]):
        super().__setstate__(state)

    # TODO: Implement
    def __getstate__(self) -> dict[str, Any]:
        superstate = super().__getstate__()
        superstate["cpu_state"] = {}
        for group in superstate["param_groups"]:
            for param in group["params"]:
                superstate["cpu_state"]
        return superstate

    def step(self, closure: Optional[Callable[[], float]] = None):

        # I'm honestly really not sure why this is part of the API.
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Step each parameter
        for group in self.param_groups:
            for p in group["params"]:
                self.step_param(p)

        return loss

    def step_param(self, param: torch.Tensor) -> None:
        param_opt = self.state.get(param)
        if type(param_opt) is not offload_adam.AdamOptimizer:
            raise ValueError("Parameter not registered with this optimizer")

        offload_adam.step(param_opt, param.data, param.grad)

    def __del__(self):
        """Free the memory held by C++. Otherwise we risk leaking unholy amounts of memory."""
        for opt in self.state.values():
            offload_adam.destroy_optimizer(opt)

    # NOTE: Not implementing zero_grad, should be handled by superclass, and
    #       doesn't require modifying optimizer state.

    @classmethod
    def vector_width(cls) -> int:
        """
        Returns 1 if using the naive scalar implementation, 256 for avx2, 512 for avx512.
        """
        return offload_adam.vector_width()
