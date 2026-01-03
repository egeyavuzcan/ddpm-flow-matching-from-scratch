"""
Flow Matching Trainer.

Handles the training loop for Flow Matching models.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict, Any
import time

from diffusion.flow_matching.probability_path import FlowMatchingPath


class FlowMatchingTrainer:
    """
    Trainer for Flow Matching models.
    
    Handles:
    - Training loop with progress bar
    - Velocity prediction loss
    - TensorBoard logging
    - Checkpoint saving/loading
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Args:
            model: UNet model
            optimizer: Optimizer (e.g., AdamW)
            device: Device to train on
            log_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        
        self.path = FlowMatchingPath()
        
        # Logging
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None
        
        # Checkpoints
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
        self.epoch = 0
    
    def train_step(
        self,
        x_0: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Single training step.
        
        Args:
            x_0: (B, C, H, W) clean images in [-1, 1]
            class_label: (B,) class labels (optional)
        
        Returns:
            Loss value
        """
        self.model.train()
        
        batch_size = x_0.shape[0]
        x_0 = x_0.to(self.device)
        if class_label is not None:
            class_label = class_label.to(self.device)
        
        # Sample timesteps (continuous [0, 1])
        t = self.path.sample_timesteps(batch_size, self.device)
        
        # Sample noise
        x_1 = torch.randn_like(x_0)
        
        # Interpolate
        x_t, _ = self.path.get_noisy_sample(x_0, t, x_1)
        
        # Target velocity
        target = x_1 - x_0
        
        # Predict velocity
        v_pred = self.model(x_t, t, class_label)
        
        # Compute loss
        loss = nn.functional.mse_loss(v_pred, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.global_step += 1
        
        return loss.item()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        log_interval: int = 100,
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            log_interval: Steps between logging
        
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                x_0, class_label = batch[0], batch[1]
            else:
                x_0, class_label = batch, None
            
            loss = self.train_step(x_0, class_label)
            total_loss += loss
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss:.4f}"})
            
            # Log to TensorBoard
            if self.writer and self.global_step % log_interval == 0:
                self.writer.add_scalar("Loss/train", loss, self.global_step)
        
        avg_loss = total_loss / num_batches
        self.epoch += 1
        
        # Log epoch summary
        if self.writer:
            self.writer.add_scalar("Loss/epoch_avg", avg_loss, self.epoch)
        
        return avg_loss
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        save_every: int = 10,
        log_interval: int = 100,
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of epochs
            save_every: Save checkpoint every N epochs
            log_interval: Steps between logging
        
        Returns:
            Training history
        """
        history = {"losses": []}
        start_time = time.time()
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader, log_interval)
            history["losses"].append(avg_loss)
            
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if self.checkpoint_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")
        
        history["training_time"] = time.time() - start_time
        
        # Save final checkpoint
        if self.checkpoint_dir:
            self.save_checkpoint("final.pt")
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        print(f"Loaded checkpoint: {path}")
