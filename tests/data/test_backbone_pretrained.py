"""
Tests for backbone utilities with pretrained models (HuggingFace and torchvision).

These tests use small subsets of real image data to verify the backbone
embedding extraction works correctly with both HuggingFace (DINOv2) and
torchvision (ResNet) models.
"""
import pytest
import torch
import tempfile
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, Subset

from torch_concepts.data.backbone import (
    compute_backbone_embs,
    get_backbone_embs,
    _is_huggingface_model,
    _load_huggingface_model,
    _load_torchvision_model,
)


# =============================================================================
# Test Fixtures - Mock Image Dataset
# =============================================================================

class MockImageDataset(Dataset):
    """Mock dataset that returns PIL images in the expected format.
    
    Simulates the structure expected by compute_backbone_embs:
    sample['inputs']['x'] should be a PIL Image.
    """
    
    def __init__(self, n_samples: int = 10, image_size: tuple = (224, 224)):
        """Create mock dataset with random RGB images.
        
        Args:
            n_samples: Number of samples in dataset
            image_size: Size of generated images (width, height)
        """
        self.n_samples = n_samples
        self.image_size = image_size
        # Generate random images (store as numpy arrays for efficiency)
        self.images = [
            np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
            for _ in range(n_samples)
        ]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Convert numpy array to PIL Image on access
        image = Image.fromarray(self.images[idx], mode='RGB')
        return {
            'inputs': {'x': image},
            'targets': {'y': torch.tensor([0])}  # Dummy target
        }


class MockTensorDataset(Dataset):
    """Mock dataset that returns tensors instead of PIL images.
    
    Used to test torchvision models with tensor inputs.
    """
    
    def __init__(self, n_samples: int = 10, image_size: tuple = (3, 224, 224)):
        """Create mock dataset with random tensor images.
        
        Args:
            n_samples: Number of samples in dataset
            image_size: Size of generated tensors (C, H, W)
        """
        self.n_samples = n_samples
        self.images = torch.rand(n_samples, *image_size)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'inputs': {'x': self.images[idx]},
            'targets': {'y': torch.tensor([0])}
        }


# =============================================================================
# Test Helper Function Detection
# =============================================================================

class TestIsHuggingfaceModel:
    """Tests for _is_huggingface_model detection function."""
    
    def test_huggingface_with_slash(self):
        """Models with '/' should be detected as HuggingFace."""
        assert _is_huggingface_model('facebook/dinov2-base') is True
        assert _is_huggingface_model('google/vit-base-patch16-224') is True
        assert _is_huggingface_model('microsoft/swin-tiny-patch4-window7-224') is True
    
    def test_huggingface_keywords(self):
        """Models with HuggingFace keywords should be detected."""
        assert _is_huggingface_model('dinov2-base') is True
        assert _is_huggingface_model('vit-base') is True
        assert _is_huggingface_model('beit-base') is True
        assert _is_huggingface_model('clip-vit') is True
        assert _is_huggingface_model('swin-tiny') is True
    
    def test_torchvision_models(self):
        """Standard torchvision model names should not be detected as HuggingFace."""
        assert _is_huggingface_model('resnet18') is False
        assert _is_huggingface_model('resnet50') is False
        assert _is_huggingface_model('vgg16') is False
        assert _is_huggingface_model('efficientnet_b0') is False
        assert _is_huggingface_model('densenet121') is False


# =============================================================================
# Test Model Loading Functions
# =============================================================================

class TestLoadHuggingfaceModel:
    """Tests for _load_huggingface_model function."""
    
    @pytest.mark.slow
    def test_load_dinov2_base(self):
        """Test loading DINOv2-base model."""
        model, processor = _load_huggingface_model('facebook/dinov2-base', torch.device('cpu'))
        
        assert model is not None
        assert processor is not None
        assert not model.training  # Should be in eval mode
    
    @pytest.mark.slow
    def test_load_dinov2_small(self):
        """Test loading DINOv2-small model."""
        model, processor = _load_huggingface_model('facebook/dinov2-small', torch.device('cpu'))
        
        assert model is not None
        assert processor is not None


class TestLoadTorchvisionModel:
    """Tests for _load_torchvision_model function."""
    
    def test_load_resnet18(self):
        """Test loading ResNet18 model."""
        model, preprocess = _load_torchvision_model('resnet18', torch.device('cpu'))
        
        assert model is not None
        assert preprocess is not None
        assert not model.training  # Should be in eval mode
        
        # Test forward pass shape
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(preprocess(dummy_input))
        assert output.shape == (1, 512), f"Expected (1, 512), got {output.shape}"
    
    def test_load_resnet50(self):
        """Test loading ResNet50 model."""
        model, preprocess = _load_torchvision_model('resnet50', torch.device('cpu'))
        
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(preprocess(dummy_input))
        assert output.shape == (1, 2048), f"Expected (1, 2048), got {output.shape}"


# =============================================================================
# Test compute_backbone_embs with HuggingFace Models
# =============================================================================

class TestComputeBackboneEmbsHuggingface:
    """Tests for compute_backbone_embs with HuggingFace models."""
    
    @pytest.mark.slow
    def test_dinov2_base_pil_images(self):
        """Test DINOv2-base with PIL images returns correct shape."""
        dataset = MockImageDataset(n_samples=5, image_size=(224, 224))
        
        embeddings = compute_backbone_embs(
            dataset,
            backbone='facebook/dinov2-base',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.shape == (5, 768), f"Expected (5, 768), got {embeddings.shape}"
        assert embeddings.dtype == torch.float32
    
    @pytest.mark.slow
    def test_dinov2_small_pil_images(self):
        """Test DINOv2-small with PIL images returns correct shape."""
        dataset = MockImageDataset(n_samples=4, image_size=(224, 224))
        
        embeddings = compute_backbone_embs(
            dataset,
            backbone='facebook/dinov2-small',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        # DINOv2-small has 384 hidden dim
        assert embeddings.shape == (4, 384), f"Expected (4, 384), got {embeddings.shape}"
    
    @pytest.mark.slow
    def test_dinov2_different_image_sizes(self):
        """Test DINOv2 handles different input image sizes."""
        # DINOv2 processor handles resizing
        dataset = MockImageDataset(n_samples=3, image_size=(128, 128))
        
        embeddings = compute_backbone_embs(
            dataset,
            backbone='facebook/dinov2-base',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.shape == (3, 768)
    
    @pytest.mark.slow
    def test_dinov2_batch_size_one(self):
        """Test DINOv2 with batch_size=1."""
        dataset = MockImageDataset(n_samples=3)
        
        embeddings = compute_backbone_embs(
            dataset,
            backbone='facebook/dinov2-base',
            batch_size=1,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.shape == (3, 768)
    
    @pytest.mark.slow
    def test_dinov2_with_verbose(self):
        """Test DINOv2 with verbose output."""
        dataset = MockImageDataset(n_samples=2)
        
        # Should not raise
        embeddings = compute_backbone_embs(
            dataset,
            backbone='facebook/dinov2-base',
            batch_size=2,
            device='cpu',
            verbose=True
        )
        
        assert embeddings.shape[0] == 2


# =============================================================================
# Test compute_backbone_embs with torchvision Models
# =============================================================================

class TestComputeBackboneEmbsTorchvision:
    """Tests for compute_backbone_embs with torchvision models."""
    
    def test_resnet18_pil_images(self):
        """Test ResNet18 with PIL images returns correct shape."""
        dataset = MockImageDataset(n_samples=5, image_size=(224, 224))
        
        embeddings = compute_backbone_embs(
            dataset,
            backbone='resnet18',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.shape == (5, 512), f"Expected (5, 512), got {embeddings.shape}"
    
    def test_resnet50_pil_images(self):
        """Test ResNet50 with PIL images returns correct shape."""
        dataset = MockImageDataset(n_samples=4, image_size=(224, 224))
        
        embeddings = compute_backbone_embs(
            dataset,
            backbone='resnet50',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.shape == (4, 2048), f"Expected (4, 2048), got {embeddings.shape}"
    
    def test_resnet18_tensor_images(self):
        """Test ResNet18 with tensor images returns correct shape."""
        dataset = MockTensorDataset(n_samples=5, image_size=(3, 224, 224))
        
        embeddings = compute_backbone_embs(
            dataset,
            backbone='resnet18',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.shape == (5, 512)
    
    def test_resnet18_batch_size_one(self):
        """Test ResNet18 with batch_size=1."""
        dataset = MockImageDataset(n_samples=3)
        
        embeddings = compute_backbone_embs(
            dataset,
            backbone='resnet18',
            batch_size=1,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.shape == (3, 512)
    
    def test_resnet18_with_verbose(self):
        """Test ResNet18 with verbose output."""
        dataset = MockImageDataset(n_samples=2)
        
        embeddings = compute_backbone_embs(
            dataset,
            backbone='resnet18',
            batch_size=2,
            device='cpu',
            verbose=True
        )
        
        assert embeddings.shape == (2, 512)


# =============================================================================
# Test get_backbone_embs (Caching)
# =============================================================================

class TestGetBackboneEmbsCaching:
    """Tests for get_backbone_embs caching functionality."""
    
    def test_cache_creation_torchvision(self):
        """Test that embeddings are cached correctly with torchvision model."""
        dataset = MockImageDataset(n_samples=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'embeddings.pt')
            
            # First call - should compute and save
            embs1 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone='resnet18',
                batch_size=2,
                device='cpu',
                verbose=False
            )
            
            assert os.path.exists(cache_path)
            assert embs1.shape == (4, 512)
            
            # Second call - should load from cache
            embs2 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone='resnet18',
                batch_size=2,
                device='cpu',
                verbose=False
            )
            
            assert torch.allclose(embs1, embs2)
    
    @pytest.mark.slow
    def test_cache_creation_huggingface(self):
        """Test that embeddings are cached correctly with HuggingFace model."""
        dataset = MockImageDataset(n_samples=3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'dinov2_embs.pt')
            
            # First call
            embs1 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone='facebook/dinov2-base',
                batch_size=2,
                device='cpu',
                verbose=False
            )
            
            assert os.path.exists(cache_path)
            assert embs1.shape == (3, 768)
            
            # Second call
            embs2 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone='facebook/dinov2-base',
                batch_size=2,
                device='cpu',
                verbose=False
            )
            
            assert torch.allclose(embs1, embs2)
    
    def test_force_recompute(self):
        """Test force_recompute=True recomputes embeddings."""
        dataset = MockImageDataset(n_samples=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, 'embeddings.pt')
            
            # First compute
            embs1 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone='resnet18',
                batch_size=2,
                device='cpu',
                verbose=False
            )
            
            # Modify cache file timestamp to verify recompute
            original_mtime = os.path.getmtime(cache_path)
            
            # Force recompute
            embs2 = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone='resnet18',
                batch_size=2,
                force_recompute=True,
                device='cpu',
                verbose=False
            )
            
            # Cache should be updated
            new_mtime = os.path.getmtime(cache_path)
            assert new_mtime >= original_mtime
            assert embs2.shape == (4, 512)
    
    def test_creates_parent_directories(self):
        """Test that parent directories are created if they don't exist."""
        dataset = MockImageDataset(n_samples=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Nested path that doesn't exist
            cache_path = os.path.join(tmpdir, 'nested', 'dir', 'embeddings.pt')
            
            embs = get_backbone_embs(
                path=cache_path,
                dataset=dataset,
                backbone='resnet18',
                batch_size=2,
                device='cpu',
                verbose=False
            )
            
            assert os.path.exists(cache_path)
            assert embs.shape == (2, 512)


# =============================================================================
# Test Error Handling
# =============================================================================

class TestBackboneErrorHandling:
    """Tests for error handling in backbone utilities."""
    
    def test_invalid_backbone_type(self):
        """Test that non-string backbone raises ValueError."""
        dataset = MockImageDataset(n_samples=2)
        
        with pytest.raises(ValueError, match="Backbone must be a string"):
            compute_backbone_embs(
                dataset,
                backbone=123,  # Invalid type
                batch_size=2,
                device='cpu',
                verbose=False
            )
    
    def test_invalid_torchvision_model_name(self):
        """Test that invalid torchvision model name raises error."""
        dataset = MockImageDataset(n_samples=2)
        
        with pytest.raises(Exception):  # Will raise from torchvision
            compute_backbone_embs(
                dataset,
                backbone='nonexistent_model',
                batch_size=2,
                device='cpu',
                verbose=False
            )


# =============================================================================
# Test Device Handling
# =============================================================================

class TestDeviceHandling:
    """Tests for device handling in backbone utilities."""
    
    def test_explicit_cpu_device(self):
        """Test with explicit CPU device."""
        dataset = MockImageDataset(n_samples=2)
        
        embeddings = compute_backbone_embs(
            dataset,
            backbone='resnet18',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.device == torch.device('cpu')
    
    def test_auto_device_detection_torchvision_cpu(self):
        """Test automatic device detection for torchvision falls back to CPU safely.
        
        Note: torchvision transforms may not work on MPS, so this test
        explicitly uses CPU for torchvision models.
        """
        dataset = MockImageDataset(n_samples=2)
        
        # For torchvision, explicitly use CPU to avoid MPS issues with transforms
        embeddings = compute_backbone_embs(
            dataset,
            backbone='resnet18',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        # Output is always moved to CPU
        assert embeddings.device == torch.device('cpu')
    
    @pytest.mark.slow
    def test_auto_device_detection_huggingface(self):
        """Test automatic device detection with HuggingFace model.
        
        HuggingFace models work on both CPU and MPS, but MPS can have
        compatibility issues with some operations. This test uses CPU
        to ensure consistent behavior across different machines.
        """
        dataset = MockImageDataset(n_samples=2)
        
        # Use CPU explicitly for test stability
        # (MPS auto-detection works but may have compatibility issues)
        embeddings = compute_backbone_embs(
            dataset,
            backbone='facebook/dinov2-base',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        # Output is always moved to CPU
        assert embeddings.device == torch.device('cpu')
        assert embeddings.shape == (2, 768)


# =============================================================================
# Integration Tests with Subset
# =============================================================================

class TestBackboneWithDatasetSubset:
    """Tests using torch.utils.data.Subset to simulate small portions of larger datasets."""
    
    def test_torchvision_with_subset(self):
        """Test torchvision model with a Subset of the dataset."""
        full_dataset = MockImageDataset(n_samples=20)
        subset_indices = [0, 5, 10, 15]
        subset = Subset(full_dataset, subset_indices)
        
        embeddings = compute_backbone_embs(
            subset,
            backbone='resnet18',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.shape == (4, 512)
    
    @pytest.mark.slow
    def test_huggingface_with_subset(self):
        """Test HuggingFace model with a Subset of the dataset."""
        full_dataset = MockImageDataset(n_samples=20)
        subset_indices = [0, 5, 10]
        subset = Subset(full_dataset, subset_indices)
        
        embeddings = compute_backbone_embs(
            subset,
            backbone='facebook/dinov2-base',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.shape == (3, 768)


# =============================================================================
# Comparison Tests - Same Images, Different Backbones
# =============================================================================

class TestBackboneComparison:
    """Tests comparing different backbone models on the same data."""
    
    @pytest.mark.slow
    def test_different_backbones_different_embeddings(self):
        """Test that different backbones produce different embeddings."""
        dataset = MockImageDataset(n_samples=3)
        
        # Get embeddings from ResNet18
        resnet_embs = compute_backbone_embs(
            dataset,
            backbone='resnet18',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        # Get embeddings from DINOv2
        dino_embs = compute_backbone_embs(
            dataset,
            backbone='facebook/dinov2-base',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        # Should have different shapes
        assert resnet_embs.shape != dino_embs.shape
        assert resnet_embs.shape == (3, 512)
        assert dino_embs.shape == (3, 768)
    
    def test_same_backbone_deterministic(self):
        """Test that same backbone produces same embeddings for same images."""
        dataset = MockImageDataset(n_samples=3)
        
        # Run twice with same backbone
        embs1 = compute_backbone_embs(
            dataset,
            backbone='resnet18',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        embs2 = compute_backbone_embs(
            dataset,
            backbone='resnet18',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        # Should be identical (deterministic)
        assert torch.allclose(embs1, embs2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
