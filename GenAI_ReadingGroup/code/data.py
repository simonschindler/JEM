import torch as t
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class MoonsDataset(Dataset):
    """
    Custom PyTorch Dataset for the scikit-learn make_moons dataset.

    This class demonstrates the standard pattern for custom datasets, which is
    essential for handling large datasets that do not fit into RAM. In the
    `__getitem__` method, data is fetched one sample at a time. For a true
    out-of-core dataset, this is where you would read data from a file on disk.
    """

    def __init__(self, n_samples=1000, noise=0.1, random_state=42):
        """
        Args:
            n_samples (int): Number of samples to generate.
            noise (float): Noise level.
            random_state (int): Random seed for reproducibility.
        """
        self.X, self.y = make_moons(
            n_samples=n_samples,
            shuffle=True,
            noise=noise,
            random_state=random_state,
        )
        self.n_samples = n_samples

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        """
        # Get the specific sample and convert to tensors on the fly
        feature = t.tensor(self.X[idx], dtype=t.float32)
        label = t.tensor(self.y[idx], dtype=t.long)
        return feature, label
