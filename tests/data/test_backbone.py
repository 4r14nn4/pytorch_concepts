"""
Tests for torch_concepts.data.backbone module.

Note: The backbone API has been updated to support only string-based model names
(HuggingFace or torchvision models). Custom nn.Module backbones are no longer
supported in the current API.

For comprehensive tests of the current backbone functionality, see:
- test_backbone_pretrained.py (tests for HuggingFace and torchvision models)
"""
import pytest
import torch
from torch_concepts.data.backbone import (
    compute_backbone_embs,
    get_backbone_embs,
    _is_huggingface_model,
)


class TestBackboneAPIValidation:
    """Tests validating the backbone API constraints."""

    def test_backbone_requires_string(self):
        """Test that backbone parameter must be a string."""
        from torch import nn
        from PIL import Image
        import numpy as np
        from torch.utils.data import Dataset
        
        class DummyDataset(Dataset):
            def __len__(self):
                return 2
            def __getitem__(self, idx):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                return {'inputs': {'x': img}}
        
        dataset = DummyDataset()
        
        # Passing nn.Module should raise ValueError
        with pytest.raises(ValueError, match="Backbone must be a string"):
            compute_backbone_embs(
                dataset,
                backbone=nn.Linear(10, 5),  # Invalid - must be string
                batch_size=2,
                device='cpu',
                verbose=False
            )
    
    def test_backbone_accepts_torchvision_string(self):
        """Test that torchvision model names are accepted."""
        from PIL import Image
        import numpy as np
        from torch.utils.data import Dataset
        
        class DummyDataset(Dataset):
            def __len__(self):
                return 2
            def __getitem__(self, idx):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                return {'inputs': {'x': img}}
        
        dataset = DummyDataset()
        
        # Should work with torchvision model name
        embeddings = compute_backbone_embs(
            dataset,
            backbone='resnet18',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.shape == (2, 512)
    
    @pytest.mark.slow
    def test_backbone_accepts_huggingface_string(self):
        """Test that HuggingFace model names are accepted."""
        from PIL import Image
        import numpy as np
        from torch.utils.data import Dataset
        
        class DummyDataset(Dataset):
            def __len__(self):
                return 2
            def __getitem__(self, idx):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                return {'inputs': {'x': img}}
        
        dataset = DummyDataset()
        
        # Should work with HuggingFace model name
        embeddings = compute_backbone_embs(
            dataset,
            backbone='facebook/dinov2-base',
            batch_size=2,
            device='cpu',
            verbose=False
        )
        
        assert embeddings.shape == (2, 768)


class TestIsHuggingfaceModelDetection:
    """Tests for the _is_huggingface_model helper function."""
    
    def test_detects_slash_as_huggingface(self):
        """Models with '/' are detected as HuggingFace."""
        assert _is_huggingface_model('facebook/dinov2-base') is True
        assert _is_huggingface_model('google/vit-base-patch16-224') is True
    
    def test_detects_keywords_as_huggingface(self):
        """Models with HuggingFace keywords are detected."""
        assert _is_huggingface_model('dinov2-base') is True
        assert _is_huggingface_model('vit-large') is True
    
    def test_torchvision_names_not_huggingface(self):
        """Standard torchvision names are not detected as HuggingFace."""
        assert _is_huggingface_model('resnet18') is False
        assert _is_huggingface_model('resnet50') is False
        assert _is_huggingface_model('vgg16') is False
        assert _is_huggingface_model('efficientnet_b0') is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
