import torch as t
import torch.nn as nn
from typing import Literal
from tqdm import tqdm

from matplotlib import pyplot as plt
import matplotlib.animation as animation


class SGLD:
    """
    Implements Stochastic Gradient Langevin Dynamics (SGLD) for sampling from an energy-based model.
    """

    def __init__(
        self,
        model: nn.Module,
        alpha=0.1,
        beta=0.9,
        gamma=0.95,
        batch_size=128,
        domain_dims=1,
        device="cpu",
        init_strategy: Literal["uniform", "gaussian"] = "uniform",
        init_range=(-1, 1),
    ) -> None:
        """
        Initializes the SGLD sampler.

        Args:
            model (nn.Module): The energy-based model to sample from.
            alpha (float): The initial step size.
            beta (float): The fraction of samples to carry over from the previous
                          batch (for persistent chain).
            gamma (float): The decay factor for the step size.
            batch_size (int): The number of samples to generate in each batch.
            domain_dims (int): The dimensionality of the sample space.
            device (str): The device to perform computations on (e.g., "cpu", "cuda").
            init_strategy (str): Strategy for initializing new chains.
                                 Options: 'uniform' or 'gaussian'.
            init_range (tuple): For 'uniform', a (min, max) tuple.
                                For 'gaussian', a (mean, std) tuple.
        """
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.domain_dims = domain_dims

        if init_strategy not in ["uniform", "gaussian"]:
            raise ValueError("init_strategy must be 'uniform' or 'gaussian'")
        self.init_strategy = init_strategy
        self.init_param1, self.init_param2 = init_range

        # Initialize the persistent chain (replay buffer) using the chosen strategy.
        self.x = self._initialize_samples(self.batch_size)

    def _initialize_samples(self, num_samples: int) -> t.Tensor:
        """
        Helper function to generate new samples based on the chosen strategy.
        """
        shape = (num_samples, self.domain_dims)
        if self.init_strategy == "uniform":
            # Sample from U(min, max)
            min_val, max_val = self.init_param1, self.init_param2
            return (max_val - min_val) * t.rand(shape, device=self.device) + min_val
        elif self.init_strategy == "gaussian":
            # Sample from N(mean, std^2)
            mean, std = self.init_param1, self.init_param2
            return t.randn(shape, device=self.device) * std + mean
        return t.zeros(shape, device=self.device)

    def reset_x(self):
        """Resets the persistent chain to a new set of random samples."""
        self.x = self._initialize_samples(self.batch_size)

    def sample(self, n_steps=20):
        """
        Generates a batch of samples using SGLD.
        """
        self.model.eval()

        num_reinit = int((1.0 - self.beta) * self.batch_size)
        if num_reinit > 0:
            rand_indices = t.randperm(self.batch_size, device=self.device)[:num_reinit]
            self.x[rand_indices, :] = self._initialize_samples(num_reinit)

        x_k = self.x.clone().requires_grad_(True)

        for i in range(n_steps):
            step_size = self.alpha * (self.gamma**i)
            grad_x = t.autograd.grad(self.model(x_k).sum(), [x_k], retain_graph=True)[0]
            noise = t.sqrt(t.tensor(step_size, device=self.device)) * t.randn_like(x_k)
            x_k.data.add_(-step_size / 2 * grad_x + noise)

        self.model.train()
        self.x = x_k.detach()

        return self.x


if __name__ == "__main__":
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def E_theta(mus, x):
        """Vectorized energy function for a Gaussian Mixture Model."""
        x_expanded = x.unsqueeze(1)
        mus_expanded = mus.unsqueeze(0)
        terms = -((x_expanded - mus_expanded) ** 2) / 2
        energy = -t.logsumexp(terms.sum(dim=2), dim=1)
        return energy

    def p_theta(mus, x):
        """Unnormalized probability density function."""
        return t.exp(-E_theta(mus, x))

    # EBM-GMM in torch
    class EBM_GMM(nn.Module):
        def __init__(self, mus):
            super().__init__()
            # Move mus to the correct device upon initialization
            self.mus = mus.to(device)

        def forward(self, x):
            return E_theta(self.mus, x)

    # --- 1. SETUP THE MODEL AND DATA ---
    mus = t.tensor([[-5.0, 0.0], [3.0, 1.0], [0.0, -4.0], [2.0, -8.0]])
    gmm = EBM_GMM(mus=mus)

    # --- 2. RUN THE SGLD SAMPLER ---
    sgld = SGLD(
        model=gmm,
        domain_dims=2,
        batch_size=2000,
        device=device,
        init_strategy="uniform",
        init_range=(-10, 10),
    )

    s = []
    print("Generating SGLD samples...")
    for i in tqdm(range(100)):
        # Detach and move to CPU for plotting
        s.append(sgld.sample(n_steps=50).cpu().clone())

    # --- 3. SETUP THE PLOT FOR ANIMATION ---
    print("Setting up animation plot...")
    # Create a 1x2 subplot figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Create grid for plotting the probability landscape
    xlims = t.tensor([-10.0, 10.0])
    ylims = t.tensor([-10.0, 10.0])
    npoints = 200

    X_coords = t.linspace(xlims[0], xlims[1], npoints)
    Y_coords = t.linspace(ylims[0], ylims[1], npoints)
    X, Y = t.meshgrid(X_coords, Y_coords, indexing="xy")

    X_grid = t.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)

    with t.no_grad():
        # Calculate both probability and energy landscapes
        probs = p_theta(mus.to(device), X_grid).reshape(npoints, npoints).cpu()
        energy = E_theta(mus.to(device), X_grid).reshape(npoints, npoints).cpu()

    # --- Plot p_theta on the left subplot (axes[0]) ---
    axes[0].contourf(X, Y, probs, levels=50, cmap="Blues")
    axes[0].plot(
        mus[:, 0], mus[:, 1], "kx", markersize=10, markeredgewidth=2, label="GMM Means"
    )
    axes[0].set_title(r"Probability Landscape $p_\theta(x)$")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].legend()
    axes[0].set_aspect("equal", adjustable="box")

    # --- Plot E_theta on the right subplot (axes[1]) ---
    axes[1].contourf(X, Y, energy, levels=50, cmap="Reds")
    axes[1].plot(
        mus[:, 0], mus[:, 1], "kx", markersize=10, markeredgewidth=2, label="GMM Means"
    )
    axes[1].set_title(r"Energy Landscape $E_\theta(x)$")
    axes[1].set_xlabel("x1")
    axes[1].legend()
    axes[1].set_aspect("equal", adjustable="box")

    # Initialize two scatter plots for the samples, one for each subplot
    scatter_p = axes[0].scatter(s[0][:, 0], s[0][:, 1], c="r", s=10, alpha=0.7)
    scatter_e = axes[1].scatter(s[0][:, 0], s[0][:, 1], c="b", s=10, alpha=0.7)
    iter_text = axes[0].text(
        0.05,
        0.95,
        "",
        transform=axes[0].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()

    # --- 4. DEFINE ANIMATION FUNCTIONS ---
    def init():
        """Initializes the animation."""
        scatter_p.set_offsets(s[0])
        scatter_e.set_offsets(s[0])
        iter_text.set_text("")
        return scatter_p, scatter_e, iter_text

    def update(frame):
        """Update function for the animation."""
        data = s[frame]
        # Update the positions of the scatter plot points on both plots
        scatter_p.set_offsets(data)
        scatter_e.set_offsets(data)
        iter_text.set_text(f"Iteration: {frame + 1}")
        return scatter_p, scatter_e, iter_text

    # --- 5. CREATE AND DISPLAY THE ANIMATION ---
    print("Creating animation...")
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(s),
        init_func=init,
        blit=True,
        interval=50,  # Delay between frames in milliseconds
    )

    # Display the animation in a live window
    plt.show()
