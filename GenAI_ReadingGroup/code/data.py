import torch
from torch.utils.data import Dataset
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class MoonsDataset(Dataset):
    """
    Custom PyTorch Dataset for the scikit-learn make_moons dataset.

    This class serves as a wrapper for the `make_moons` function from scikit-learn,
    making it compatible with PyTorch's DataLoader. It demonstrates the standard
    pattern for creating custom datasets, which is essential for handling large
    datasets that do not fit into RAM.

    In the `__getitem__` method, data is fetched one sample at a time. For a true
    out-of-core dataset, this is where you would typically read data from a file
    on disk.
    """

    def __init__(self, n_samples: int = 1000, noise: float = 0.1, random_state: int = 42):
        """
        Initializes the dataset by generating the moon-shaped data.

        Args:
            n_samples (int): The total number of points to generate.
            noise (float): Standard deviation of Gaussian noise added to the data.
            random_state (int): Determines random number generation for dataset creation.
        """
        self.X, self.y = make_moons(
            n_samples=n_samples,
            shuffle=True,
            noise=noise,
            random_state=random_state,
        )
        self.n_samples = n_samples

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A tuple containing the feature tensor and the label tensor.
        """
        # Convert numpy arrays to PyTorch tensors on the fly
        feature = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return feature, label
