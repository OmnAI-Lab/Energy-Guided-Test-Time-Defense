import torch
import torch.nn as nn


def energy_x(logits, t: float = 1.0):
    """
    Energy score:
        E(x) = -t * logsumexp(logits / t)
    """
    return -t * torch.logsumexp(logits / t, dim=1)


class EnergytransformWrapper(nn.Module):

    def __init__(
        self,
        model,
        alpha: float = 5.0,
        eps: float = 10.0,
        t: float = 1.0,
        max_iter: int = 1,
    ):
        super().__init__()

        self.model = model
        self.alpha = alpha
        self.eps = eps
        self.t = t
        self.max_iter = max_iter

    @torch.enable_grad()
    def transform_x(self, x):
        # with torch.set_grad_enabled(True):
        delta = torch.zeros_like(x, requires_grad=True)
        final_x = x

        for _ in range(self.max_iter):
            logits = self.model(x + delta)
            energy_vec = energy_x(logits, t=self.t)

            grad = torch.autograd.grad(
                energy_vec.sum(),
                delta,
                retain_graph=False,
                create_graph=False,
            )[0]

            grad_norm = torch.norm(
                grad,
                p=2,
                dim=(1, 2, 3),
                keepdim=True,
            )

            
            delta.data -= self.alpha * grad.data / (grad_norm.data + 1e-12)

            delta.data = (x + delta.data).clamp(0.0, 1.0) - x

            delta_norm = torch.norm(
                delta.data.view(delta.data.size(0), -1),
                p=2,
                dim=1,
                keepdim=True,
            ).view(-1, 1, 1, 1)

            factor = torch.clamp(
                self.eps / (delta_norm + 1e-8),
                max=1.0,
            )

            delta.data = factor * delta.data
            final_x = torch.clamp(x + delta.data, 0.0, 1.0)

            delta = delta.detach()
            delta.requires_grad = True

        return final_x

    def forward(self, x):
        transformed_x = self.transform_x(x)
        logits = self.model(transformed_x)
        return logits