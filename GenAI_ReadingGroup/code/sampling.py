import torch
import torch.nn as nn
from typing import Literal, Tuple
from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# --- 1. SGLD Sampler Implementation ---


class SGLD:
    """
    Implements the Stochastic Gradient Langevin Dynamics (SGLD) sampler.

    SGLD is a sampling method from the family of Markov Chain Monte Carlo (MCMC)
    algorithms. It is designed to draw samples from a probability distribution
    p(x) which is known up to a normalization constant, typical for Energy-Based
    Models where p(x) ∝ exp(-E(x)).

    The update rule for SGLD at step k is:
    $$
    x_{k+1} = x_k - \\frac{\\alpha_k}{2} \\nabla_x E(x_k) + \\sqrt{\\alpha_k} \\epsilon_k
    $$
    where:
    - $x_k$ is the sample at step k.
    - $\\alpha_k$ is the step size (learning rate).
    - $\\nabla_x E(x_k)$ is the gradient of the energy function w.r.t. the sample x.
    - $\\epsilon_k$ is Gaussian noise, $\\epsilon_k \\sim \\mathcal{N}(0, I)$.
    """

    def __init__(
        self,
        model: nn.Module,
        domain_dims: int,
        batch_size: int = 128,
        device: str = "cpu",
        alpha: float = 0.1,
        gamma: float = 0.95,
        replay_prob: float = 0.95,
        init_strategy: Literal["uniform", "gaussian"] = "uniform",
        init_params: Tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """
        Initializes the SGLD sampler.

        Args:
            model (nn.Module): The energy-based model. Its forward pass should
                return the energy of the input samples.
            domain_dims (int): The dimensionality of the sample space (e.g., 2 for 2D points).
            batch_size (int): The number of parallel chains (samples) to run.
            device (str): The device to perform computations on ("cpu", "cuda").
            alpha (float): The initial step size for the Langevin dynamics.
            gamma (float): The decay factor for the step size scheduler ($\alpha_k = \alpha \cdot \gamma^k$).
            replay_prob (float): The probability of re-using a sample from the
                replay buffer vs. re-initializing it.
            init_strategy (str): Strategy for initializing new chains ('uniform' or 'gaussian').
            init_params (tuple): For 'uniform', this is (min, max). For 'gaussian',
                this is (mean, std).
        """
        self.model = model
        self.domain_dims = domain_dims
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.replay_prob = replay_prob

        if init_strategy not in ["uniform", "gaussian"]:
            raise ValueError("init_strategy must be 'uniform' or 'gaussian'")
        self.init_strategy = init_strategy
        self.init_params = init_params

        # Initialize the persistent chain (also known as a replay buffer).
        self.replay_buffer = self._initialize_samples(self.batch_size)

    def _initialize_samples(self, num_samples: int) -> torch.Tensor:
        """Helper to generate new samples based on the initialization strategy."""
        shape = (num_samples, self.domain_dims)
        if self.init_strategy == "uniform":
            min_val, max_val = self.init_params
            return (max_val - min_val) * torch.rand(shape, device=self.device) + min_val
        elif self.init_strategy == "gaussian":
            mean, std = self.init_params
            return torch.randn(shape, device=self.device) * std + mean
        # Fallback, though validation should prevent this.
        return torch.zeros(shape, device=self.device)

    def sample(self, n_steps: int = 20) -> torch.Tensor:
        """
        Generates a batch of samples by running the SGLD chain for n_steps.

        Args:
            n_steps (int): The number of SGLD steps to perform.

        Returns:
            torch.Tensor: A tensor of generated samples of shape (batch_size, domain_dims).
        """
        self.model.eval()

        # Decide which samples in the buffer to reinitialize.
        num_reinit = int((1.0 - self.replay_prob) * self.batch_size)
        if num_reinit > 0:
            reinit_indices = torch.randperm(self.batch_size, device=self.device)[
                :num_reinit
            ]
            self.replay_buffer[reinit_indices] = self._initialize_samples(num_reinit)

        # Create a new tensor for the current chain, enabling gradient tracking.
        samples = self.replay_buffer.clone().requires_grad_(True)

        for i in range(n_steps):
            step_size = self.alpha * (self.gamma**i)

            # Calculate the energy and gradients.
            energy = -self.model(samples)

            # For conditional/multi-class models, the energy is the log-sum-exp
            # of the class-conditional energies to get the marginal energy.
            if energy.ndim > 1 and energy.shape[1] > 1:
                energy = -torch.logsumexp(-energy, dim=-1)

            (grad,) = torch.autograd.grad(energy.sum(), [samples], retain_graph=True)

            # SGLD update step: gradient descent + Gaussian noise.
            noise = torch.sqrt(torch.tensor(step_size)) * torch.randn_like(samples)
            samples.data.add_(-0.5 * step_size * grad + noise)

        self.model.train()
        # Update the replay buffer with the new samples for the next round.
        self.replay_buffer = samples.detach()

        return self.replay_buffer



# --- 2. Energy-Based Model (GMM) Implementation ---


class GaussianMixtureEBM(nn.Module):
    """
    An Energy-Based Model representing a Gaussian Mixture Model.

    This model can represent either:
    1. An unconditional GMM: E(x) is the energy of a single mixture.
    2. A conditional GMM: E(x) is a vector of energies, one for each class/mixture.
    """

    def __init__(self, mus: torch.Tensor):
        """
        Args:
            mus (torch.Tensor): The means of the Gaussian components.
                - For unconditional model: shape (n_means, n_dims)
                - For conditional model: shape (n_classes, n_means_per_class, n_dims)
        """
        super().__init__()
        self.mus = nn.Parameter(mus, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the energy E(x) for a batch of samples.

        The energy is based on the negative log-likelihood of a GMM, ignoring
        normalization constants.
        Energy E(x) = -log(Σ_i exp(-||x - μ_i||² / 2))

        Args:
            x (torch.Tensor): Input samples, shape (batch_size, n_dims).

        Returns:
            torch.Tensor: Energy values.
                - Unconditional: shape (batch_size,)
                - Conditional: shape (batch_size, n_classes)
        """
        # Reshape inputs for vectorized computation:
        # x: (batch_size, 1, 1, n_dims) or (batch_size, 1, n_dims)
        # mus: (1, n_classes, n_means, n_dims) or (1, n_means, n_dims)
        x_expanded = x.unsqueeze(1)
        mus_expanded = self.mus.unsqueeze(0)
        if self.mus.ndim > 2:  # Conditional case
            x_expanded = x_expanded.unsqueeze(1)

        # Calculate squared Euclidean distance
        # terms shape: (batch_size, n_classes, n_means) or (batch_size, n_means)
        log_probs = -0.5 * torch.sum((x_expanded - mus_expanded) ** 2, dim=-1)
        # aggregate per class -> shape: (batch_size,n_classes)
        log_probs = torch.logsumexp(log_probs, dim=-1)
        return log_probs


# --- 3. Main Execution Block ---


def setup_model_and_sampler(config: dict, device: str) -> Tuple[nn.Module, SGLD]:
    """Initializes the EBM and SGLD sampler based on the configuration."""
    if config["USE_CONDITIONAL_GMM"]:
        mus1 = torch.tensor([[-5.0, 0.0], [3.0, 1.0]])
        mus2 = torch.tensor([[0.0, -4.0], [2.0, -8.0]])
        mus = torch.stack((mus1, mus2))  # Shape: (2, 2, 2)
    else:
        mus = torch.tensor(
            [[-5.0, 0.0], [3.0, 1.0], [0.0, -4.0], [2.0, -8.0]]
        )  # Shape: (4, 2)

    gmm_ebm = GaussianMixtureEBM(mus=mus).to(device)

    sgld_sampler = SGLD(
        model=gmm_ebm,
        domain_dims=config["DOMAIN_DIMS"],
        batch_size=config["BATCH_SIZE"],
        device=device,
        init_strategy="uniform",
        init_params=(-10, 10),
    )
    return gmm_ebm, sgld_sampler


def run_sampling(sampler: SGLD, config: dict) -> list:
    """Runs the SGLD sampler and collects the history of samples."""
    print("Generating SGLD samples...")
    sample_history = []
    for _ in tqdm(range(config["ANIMATION_FRAMES"])):
        samples = sampler.sample(n_steps=config["SGLD_STEPS_PER_FRAME"])
        sample_history.append(samples.cpu().clone())
    return sample_history


def create_animation(
    gmm_ebm: nn.Module, sample_history: list, config: dict, device: str
):
    """Creates and displays an animation of the sampling process."""
    print("Setting up animation plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Create grid for plotting the probability/energy landscape
    x_coords = torch.linspace(-10, 10, 200)
    y_coords = torch.linspace(-10, 10, 200)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="xy")
    grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1).to(device)

    # Calculate energy and probability landscapes
    with torch.no_grad():
        gmm_ebm.eval()
        energy_grid = gmm_ebm(grid)
        # For conditional model, get marginal energy
        if energy_grid.ndim > 1 and energy_grid.shape[1] > 1:
            energy_grid = -torch.logsumexp(energy_grid, dim=-1)
        energy_grid = energy_grid.reshape(200, 200).cpu()
        prob_grid = torch.exp(-energy_grid)

    # Define colors for the GMM means if using conditional model
    # You can customize these colors
    mean_colors = ["green", "purple", "orange", "cyan", "magenta", "yellow"]
    n_classes = gmm_ebm.mus.shape[0] if gmm_ebm.mus.ndim > 2 else 1

    # Plot landscapes
    for i, (title, data, cmap) in enumerate(
        [
            (
                r"Probability Landscape $p_\theta(x) \propto \exp(-E_\theta(x))$",
                prob_grid,
                "Blues",
            ),
            (r"Energy Landscape $E_\theta(x)$", energy_grid, "Reds"),
        ]
    ):
        axes[i].contourf(grid_x, grid_y, data, levels=50, cmap=cmap)

        if config["USE_CONDITIONAL_GMM"]:
            # Iterate through classes and plot their means with different colors
            for class_idx in range(n_classes):
                class_mus = gmm_ebm.mus[class_idx].cpu().numpy()
                axes[i].plot(
                    class_mus[:, 0],
                    class_mus[:, 1],
                    marker="x",
                    markersize=10,
                    markeredgewidth=2,
                    linestyle="None",
                    color=mean_colors[
                        class_idx % len(mean_colors)
                    ],  # Cycle through defined colors
                    label=f"Class {class_idx + 1} Means",
                )
        else:
            # Unconditional case, plot all means with a single color
            mus_cpu = gmm_ebm.mus.cpu().view(-1, 2).numpy()
            axes[i].plot(
                mus_cpu[:, 0],
                mus_cpu[:, 1],
                "kx",
                markersize=10,
                markeredgewidth=2,
                label="GMM Means",
            )

        axes[i].set_title(title)
        axes[i].set_xlabel("$x_1$")
        axes[i].set_ylabel("$x_2$")
        axes[i].legend()
        axes[i].set_aspect("equal", "box")

    # Setup scatter plots for animation
    scatter_p = axes[0].scatter([], [], c="r", s=10, alpha=0.7, label="SGLD Samples")
    scatter_e = axes[1].scatter([], [], c="b", s=10, alpha=0.7, label="SGLD Samples")
    iter_text = axes[0].text(
        0.05,
        0.95,
        "",
        transform=axes[0].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()

    # Animation functions
    def init():
        """Initializes the animation with empty 2D data."""
        empty_offsets = np.empty((0, 2))
        scatter_p.set_offsets(empty_offsets)
        scatter_e.set_offsets(empty_offsets)
        iter_text.set_text("")
        return scatter_p, scatter_e, iter_text

    def update(frame):
        """Update function for the animation."""
        data = sample_history[frame]
        scatter_p.set_offsets(data)
        scatter_e.set_offsets(data)
        iter_text.set_text(f"Frame: {frame + 1}/{len(sample_history)}")
        return scatter_p, scatter_e, iter_text

    print("Creating animation...")
    ani = animation.FuncAnimation(
        fig, update, frames=len(sample_history), init_func=init, blit=True, interval=50
    )
    plt.show()


if __name__ == "__main__":
    # --- Configuration ---
    CONFIG = {
        "USE_CONDITIONAL_GMM": False,  # <--- SET THIS TO TRUE
        "DOMAIN_DIMS": 2,
        "BATCH_SIZE": 2000,
        "ANIMATION_FRAMES": 100,
        "SGLD_STEPS_PER_FRAME": 50,
    }

    # --- Main Execution ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    model, sampler = setup_model_and_sampler(CONFIG, DEVICE)
    history = run_sampling(sampler, CONFIG)
    create_animation(model, history, CONFIG, DEVICE)
