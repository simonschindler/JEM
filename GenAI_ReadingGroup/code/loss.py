import torch as t
import torch.nn as nn
from sampling import SGLD


class p_x_loss(nn.Module):
    def __init__(self, sgld: SGLD, sample_steps=20):
        super().__init__()
        self.sgld = sgld
        self.f = sgld.model
        self.sample_steps = sample_steps

    def forward(self, x):
        # sample from the model with SGLD
        x_sample = self.sgld.sample(n_steps=self.sample_steps)
        # compute mean energy of x and of the sample
        e_sample = self.f(x_sample).mean()
        e_x = self.f(x).mean()
        loss = (e_sample - e_x)
        return loss


class p_x_y_loss(nn.Module):
    def __init__(self, sgld: SGLD, lam=0.5):
        super().__init__()
        self.p_x_loss = p_x_loss(sgld)
        self.f = sgld.model
        self.x_entropy = nn.CrossEntropyLoss()
        self.lam = lam

    def forward(self, x, y):
        x_entropy = self.x_entropy(self.f(x), y)
        p_x_loss = self.p_x_loss(x)
        return self.lam * p_x_loss + t.Tensor([
            1 - self.lam]
        ) * x_entropy, x_entropy, p_x_loss
