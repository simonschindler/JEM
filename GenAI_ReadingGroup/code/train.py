import torch
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
        optim: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optim = optim
        self.loss_fn = loss_fn
        self.device = device
        self.epoch = 0

    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
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
            A tuple containing the average loss and a classification report.
        """
        self.model.eval()  # Set the model to evaluation mode
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():  # Disable gradient computation
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                # Forward pass
                y_pred_logits = self.model(x_batch)

                # Calculate loss
                loss = self.loss_fn(y_pred_logits, y_batch)
                total_loss += loss.item()

                # Get predictions from logits
                preds = torch.argmax(y_pred_logits, dim=1)

                # Store predictions and true labels
                all_preds.append(preds.cpu())
                all_labels.append(y_batch.cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Calculate average loss
        avg_loss = total_loss / len(dataloader)

        # Generate classification report
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
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Loads the model and optimizer state from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        print(f"Checkpoint loaded from {path}")
        return checkpoint


class JEM_Training(Training):
    """
    A specialized training class for Joint Energy Models (JEM).
    """

    def train_step(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        """
        Performs a single training step for the JEM model.
        """
        self.model.train()
        x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

        # The loss function for JEM returns three values:
        # total_loss, cross_entropy_loss, and generative_loss
        loss = self.loss_fn(x_batch, y_batch)[0]

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Update the model in the SGLD sampler to ensure it uses the latest weights
        self.loss_fn.p_x_loss.sgld.model = self.model

        return loss.item()

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float, float, dict]:
        """
        Evaluates the JEM model on a given dataset.

        Returns:
            A tuple containing the average total loss, generative loss, cross-entropy
            loss, and a classification report.
        """
        self.model.eval()
        total_loss = 0.0
        total_p_x_loss = 0.0
        total_x_ent_loss = 0.0
        all_preds = []
        all_labels = []

  
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

            # Forward pass to get losses
            loss, p_x_loss, x_ent_loss = self.loss_fn(x_batch, y_batch)
            total_loss += loss.item()
            total_x_ent_loss += x_ent_loss.item()
            total_p_x_loss += p_x_loss.item()

            # Forward pass to get predictions
            y_pred_logits = self.model(x_batch)
            preds = torch.argmax(y_pred_logits, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Calculate average losses
        avg_loss = total_loss / len(dataloader)
        avg_x_ent_loss = total_x_ent_loss / len(dataloader)
        avg_p_x_loss = total_p_x_loss / len(dataloader)

        # Generate classification report
        metrics = classification_report(
            all_labels.numpy(), all_preds.numpy(), output_dict=True, zero_division=0
        )

        return avg_loss, avg_p_x_loss, avg_x_ent_loss, metrics
