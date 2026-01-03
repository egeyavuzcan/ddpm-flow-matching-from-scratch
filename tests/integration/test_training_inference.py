"""
Integration tests for training and inference pipeline.

Includes:
- Training step tests
- Checkpoint save/load tests
- End-to-end train → save → load → inference tests
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import tempfile
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from modeling.unet.unet import UNet, UNetSmall
from diffusion.ddpm.noise_schedule import NoiseSchedule
from training.ddpm_trainer import DDPMTrainer
from training.flow_matching_trainer import FlowMatchingTrainer
from inference.ddpm_sampler import DDPMSampler
from inference.flow_matching_sampler import FlowMatchingSampler


def create_dummy_dataloader(num_samples: int = 32, batch_size: int = 8):
    """Create a dummy dataloader for testing."""
    # Fake images in [-1, 1] and labels 0-9
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TestDDPMTrainer:
    """Tests for DDPM trainer."""
    
    @pytest.fixture
    def setup(self):
        """Create model, optimizer, schedule for training."""
        device = torch.device("cpu")
        model = UNetSmall(num_classes=10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        schedule = NoiseSchedule(num_timesteps=100)  # Small for testing
        return model, optimizer, schedule, device
    
    def test_train_step(self, setup):
        """Test single training step."""
        model, optimizer, schedule, device = setup
        trainer = DDPMTrainer(model, optimizer, schedule, device)
        
        x = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        
        loss = trainer.train_step(x, labels)
        
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_train_epoch(self, setup):
        """Test training for one epoch."""
        model, optimizer, schedule, device = setup
        trainer = DDPMTrainer(model, optimizer, schedule, device)
        
        dataloader = create_dummy_dataloader(num_samples=16, batch_size=4)
        
        avg_loss = trainer.train_epoch(dataloader)
        
        assert isinstance(avg_loss, float)
        assert avg_loss > 0
        assert trainer.epoch == 1
    
    def test_checkpoint_save_load(self, setup):
        """Test checkpoint saving and loading."""
        model, optimizer, schedule, device = setup
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = DDPMTrainer(
                model, optimizer, schedule, device,
                checkpoint_dir=tmpdir,
            )
            
            # Train a bit
            x = torch.randn(4, 3, 32, 32)
            labels = torch.randint(0, 10, (4,))
            trainer.train_step(x, labels)
            trainer.epoch = 5
            
            # Save
            trainer.save_checkpoint("test.pt")
            
            assert Path(tmpdir, "test.pt").exists()
            
            # Create new trainer and load
            model2 = UNetSmall(num_classes=10)
            optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
            trainer2 = DDPMTrainer(model2, optimizer2, schedule, device)
            
            trainer2.load_checkpoint(Path(tmpdir, "test.pt"))
            
            assert trainer2.epoch == 5


class TestFlowMatchingTrainer:
    """Tests for Flow Matching trainer."""
    
    @pytest.fixture
    def setup(self):
        """Create model, optimizer for training."""
        device = torch.device("cpu")
        model = UNetSmall(num_classes=10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        return model, optimizer, device
    
    def test_train_step(self, setup):
        """Test single training step."""
        model, optimizer, device = setup
        trainer = FlowMatchingTrainer(model, optimizer, device)
        
        x = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        
        loss = trainer.train_step(x, labels)
        
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_train_epoch(self, setup):
        """Test training for one epoch."""
        model, optimizer, device = setup
        trainer = FlowMatchingTrainer(model, optimizer, device)
        
        dataloader = create_dummy_dataloader(num_samples=16, batch_size=4)
        
        avg_loss = trainer.train_epoch(dataloader)
        
        assert isinstance(avg_loss, float)
        assert avg_loss > 0


class TestDDPMSampler:
    """Tests for DDPM sampler."""
    
    def test_sample_shape(self):
        """Test sample output shape."""
        device = torch.device("cpu")
        model = UNetSmall(num_classes=10)
        schedule = NoiseSchedule(num_timesteps=10)  # Very small for speed
        sampler = DDPMSampler(model, schedule, device)
        
        samples = sampler.sample(
            batch_size=2,
            image_size=32,
            class_label=torch.tensor([0, 5]),
            show_progress=False,
        )
        
        assert samples.shape == (2, 3, 32, 32)
    
    def test_sample_trajectory(self):
        """Test sampling with trajectory."""
        device = torch.device("cpu")
        model = UNetSmall(num_classes=10)
        schedule = NoiseSchedule(num_timesteps=10)
        sampler = DDPMSampler(model, schedule, device)
        
        samples, trajectory = sampler.sample_with_trajectory(
            batch_size=1,
            save_every=5,
        )
        
        assert samples.shape == (1, 3, 32, 32)
        assert len(trajectory) > 1


class TestFlowMatchingSampler:
    """Tests for Flow Matching sampler."""
    
    def test_sample_shape_euler(self):
        """Test sample output shape with Euler solver."""
        device = torch.device("cpu")
        model = UNetSmall(num_classes=10)
        sampler = FlowMatchingSampler(model, device)
        
        samples = sampler.sample(
            batch_size=2,
            image_size=32,
            num_steps=5,  # Very small for speed
            class_label=torch.tensor([0, 5]),
            show_progress=False,
            solver="euler",
        )
        
        assert samples.shape == (2, 3, 32, 32)
    
    def test_sample_shape_heun(self):
        """Test sample output shape with Heun solver."""
        device = torch.device("cpu")
        model = UNetSmall(num_classes=10)
        sampler = FlowMatchingSampler(model, device)
        
        samples = sampler.sample(
            batch_size=2,
            image_size=32,
            num_steps=5,
            class_label=torch.tensor([0, 5]),
            show_progress=False,
            solver="heun",
        )
        
        assert samples.shape == (2, 3, 32, 32)


class TestEndToEndPipeline:
    """End-to-end tests: train → save → load → inference."""
    
    def test_ddpm_end_to_end(self):
        """Test full DDPM pipeline."""
        device = torch.device("cpu")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create model and train
            model = UNetSmall(num_classes=10)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            schedule = NoiseSchedule(num_timesteps=10)
            
            trainer = DDPMTrainer(
                model, optimizer, schedule, device,
                checkpoint_dir=tmpdir,
            )
            
            dataloader = create_dummy_dataloader(num_samples=8, batch_size=4)
            trainer.train(dataloader, num_epochs=1, save_every=1)
            
            # 2. Verify checkpoint exists
            checkpoint_path = Path(tmpdir, "final.pt")
            assert checkpoint_path.exists()
            
            # 3. Load model and sample
            model2 = UNetSmall(num_classes=10)
            model2.load_state_dict(
                torch.load(checkpoint_path)["model_state_dict"]
            )
            
            sampler = DDPMSampler(model2, schedule, device)
            samples = sampler.sample(
                batch_size=2,
                class_label=torch.tensor([0, 1]),
                show_progress=False,
            )
            
            assert samples.shape == (2, 3, 32, 32)
    
    def test_flow_matching_end_to_end(self):
        """Test full Flow Matching pipeline."""
        device = torch.device("cpu")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create model and train
            model = UNetSmall(num_classes=10)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            trainer = FlowMatchingTrainer(
                model, optimizer, device,
                checkpoint_dir=tmpdir,
            )
            
            dataloader = create_dummy_dataloader(num_samples=8, batch_size=4)
            trainer.train(dataloader, num_epochs=1, save_every=1)
            
            # 2. Verify checkpoint exists
            checkpoint_path = Path(tmpdir, "final.pt")
            assert checkpoint_path.exists()
            
            # 3. Load model and sample
            model2 = UNetSmall(num_classes=10)
            model2.load_state_dict(
                torch.load(checkpoint_path)["model_state_dict"]
            )
            
            sampler = FlowMatchingSampler(model2, device)
            samples = sampler.sample(
                batch_size=2,
                num_steps=5,
                class_label=torch.tensor([0, 1]),
                show_progress=False,
            )
            
            assert samples.shape == (2, 3, 32, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
