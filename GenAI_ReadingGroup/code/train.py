import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report  # For detailed metrics


class Training:
    """
    A class to handle the training and evaluation process of a PyTorch model.
    It includes methods for a single training step, a full epoch, evaluation,
    and for saving/loading checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        optim: t.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optim = optim
        self.loss_fn = loss_fn
        self.device = device
        self.epoch = 0

    def train_step(self, x_batch: t.Tensor, y_batch: t.Tensor) -> float:
        """
        Performs a single training step: forward pass, loss calculation,
        backward pass, and optimizer step.
        """
        self.model.train()
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

        y_pred = self.model(x_batch)
        loss = self.loss_fn(y_pred, y_batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Trains the model for one full epoch.

        Args:
            dataloader (DataLoader): The DataLoader providing the training data.

        Returns:
            float: The average loss over all batches in the epoch.
        """
        self.model.train()
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            loss = self.train_step(x_batch, y_batch)
            total_loss += loss

        self.epoch += 1
        return total_loss / len(dataloader)

    def evaluate(self, dataloader: DataLoader) -> tuple[float, dict]:
        """
        Evaluates the model on a given dataset.

        Args:
            dataloader (DataLoader): The DataLoader providing the evaluation data.

        Returns:
            tuple[float, dict]: A tuple containing the average loss and a dictionary
                                with classification metrics (precision, recall, f1-score).
        """
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with t.no_grad():  # Disable gradient calculations
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                # Forward pass
                y_pred_logits = self.model(x_batch)

                # Calculate loss
                loss = self.loss_fn(y_pred_logits, y_batch)
                total_loss += loss.item()

                # Get predictions from logits
                preds = t.argmax(y_pred_logits, dim=1)

                # Store predictions and true labels
                all_preds.append(preds.cpu())
                all_labels.append(y_batch.cpu())

        # Concatenate all batches
        all_preds = t.cat(all_preds)
        all_labels = t.cat(all_labels)

        # Calculate average loss
        avg_loss = total_loss / len(dataloader)

        # Generate classification report
        # output_dict=True makes it easy to parse the results later
        metrics = classification_report(
            all_labels.numpy(), all_preds.numpy(), output_dict=True, zero_division=0
        )

        return avg_loss, metrics

    def save_checkpoint(self, path: str, **kwargs):
        """Saves the model and optimizer state to a file."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
        }
        checkpoint.update(kwargs)
        t.save(checkpoint, path)
        print(f"\nCheckpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Loads the model and optimizer state from a file."""
        checkpoint = t.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from {path}")
        return checkpoint
