import torch as t
import torch.nn as nn


class SGLD:
    """
    Class implementing Stochastic Gradient Langevin Dynamics (SGLD)
    with Persistent Contrastive Divergence (PCD) for faster sampling
    convergence. Be aware that PCD induces a bias which might lead to instable training.
    Turn PCD of by setting beta=0.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha=0.1,
        beta=0.9,
        gamma=1 / 2,
        batch_size=128,
        domain_dims=1,
        device="cpu",
    ) -> None:
        """
        Args:
            alpha: steps size, noise standard deviation
            beta: fraction of samples used from last iteration when sampling a new batch.
        """
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.domain_dims = domain_dims
        self.x = t.rand(size=[batch_size, domain_dims], device=device)

    def reset_x(self):
        self.x = t.rand(size=[self.batch_size, self.domain_dims], device=self.device)

    def sample(self, n_steps=20):
        self.model.eval()
        # reset (1-beta)*batch_size entries in x to come from a uniform dist
        rand_ind = t.randperm(self.batch_size)[: int(self.beta * self.batch_size)]
        self.x[rand_ind, :] = t.rand(
            size=[int(self.beta * self.batch_size), self.domain_dims],
            device=self.device,
        )
        x_k = t.autograd.Variable(self.x, requires_grad=True)
        for i in range(n_steps):
            # produce some noise baby
            eps = t.pow(self.gamma, t.Tensor(i)) * self.alpha * t.randn_like(self.x)
            # gather gradient
            grad_x = t.autograd.grad(self.model(x_k).sum(), [x_k], retain_graph=True)[0]
            x_k.data += t.pow(self.gamma, t.Tensor(i)) * self.alpha / 2 * grad_x + eps

        self.model.train()
        # store into replay buffeer
        self.x = x_k.detach()

        return self.x
