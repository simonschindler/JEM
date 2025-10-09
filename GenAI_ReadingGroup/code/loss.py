import torch
import torch.nn as nn
from sampling import SGLD


class PxLoss(nn.Module):
    """
    Computes the loss for the generative model, which is defined as the difference
    between the energy of samples from the data distribution and samples from the
    model distribution.
    """

    def __init__(self, sgld: SGLD, sample_steps: int = 20):
        """
        Initializes the loss module.

        Args:
            sgld (SGLD): The SGLD sampler used to generate samples from the model.
            sample_steps (int): The number of steps to run the SGLD sampler for.
        """
        super().__init__()
        self.sgld = sgld
        self.f = sgld.model
        self.sample_steps = sample_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss.

        Args:
            x (torch.Tensor): A batch of samples from the data distribution.

        Returns:
            The loss value.
        """
        # Sample from the model distribution using SGLD
        x_sample = self.sgld.sample(n_steps=self.sample_steps)

        # Compute the mean energy of the data samples and the generated samples
        e_x = self.f(x).mean()
        e_sample = self.f(x_sample).mean()

        # The loss is the difference between the two energies
        loss = e_sample - e_x
        return loss


class PxYLoss(nn.Module):
    """
    Computes the combined loss for the Joint Energy Model (JEM), which includes
    both a generative loss and a discriminative (cross-entropy) loss.
    """

    def __init__(self, sgld: SGLD, lam: float = 0.5):
        """
        Initializes the loss module.

        Args:
            sgld (SGLD): The SGLD sampler used for the generative loss.
            lam (float): The weight for the generative loss component.
        """
        super().__init__()
        self.p_x_loss = PxLoss(sgld)
        self.f = sgld.model
        self.x_entropy = nn.CrossEntropyLoss()
        self.lam = lam

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the combined loss.

        Args:
            x (torch.Tensor): A batch of input samples.
            y (torch.Tensor): The corresponding labels.

        Returns:
            A tuple containing the total loss, the cross-entropy loss, and the
            generative loss.
        """
        # Discriminative loss
        x_entropy = self.x_entropy(self.f(x), y)

        # Generative loss
        p_x_loss = self.p_x_loss(x)

        # Combined loss
        total_loss = self.lam * p_x_loss + (1 - self.lam) * x_entropy
        return total_loss, x_entropy, p_x_loss
