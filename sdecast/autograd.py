import torch
from torch import nn, Tensor

@torch.compiler.disable
def jvp(f, x: Tensor, v: Tensor) -> tuple[Tensor, ...]:
    return torch.autograd.functional.jvp(
        f, x, v,
        create_graph=torch.is_grad_enabled()
    )

def t_dir(f, t: Tensor) -> tuple[Tensor, ...]:
    return jvp(f, t, torch.ones_like(t))

@torch.compiler.disable
def grad(f, x: Tensor) -> tuple[Tensor, Tensor]:
    create_graph = torch.is_grad_enabled()

    with torch.enable_grad():
        x = x.clone()

        if not x.requires_grad:
            x.requires_grad = True

        y = f(x)

        (gradient, ) = torch.autograd.grad(y.sum(), x, create_graph=create_graph)

    return y, gradient
