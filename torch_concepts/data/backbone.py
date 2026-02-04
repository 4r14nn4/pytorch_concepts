"""Backbone utilities for feature extraction and embedding precomputation.

Provides functions to extract and cache embeddings from pre-trained backbone
models (e.g., DINOv2, ResNet, ViT) to speed up training of concept-based models.
"""
import os
import torch
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _load_huggingface_model(backbone: str, device: torch.device):
    """Load a HuggingFace model and processor."""
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone).to(device).eval()
    return model, processor


def _load_torchvision_model(backbone: str, device: torch.device):
    """Load a torchvision model and its preprocessing transforms."""
    from torchvision.models import get_model, get_model_weights
    import torch.nn as nn
    
    weights = get_model_weights(backbone).DEFAULT
    full_model = get_model(backbone, weights=weights)
    
    # Remove the classification head to get embeddings
    backbone_lower = backbone.lower()
    if 'resnet' in backbone_lower:
        # ResNet: remove final FC layer, keep avgpool
        model = nn.Sequential(*list(full_model.children())[:-1], nn.Flatten())
    elif 'vgg' in backbone_lower:
        # VGG: use features + avgpool
        model = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten())
    elif 'efficientnet' in backbone_lower:
        # EfficientNet: remove classifier
        model = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten())
    elif 'densenet' in backbone_lower:
        # DenseNet: use features + avgpool + flatten
        model = nn.Sequential(full_model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    else:
        # Generic: try removing last layer
        model = nn.Sequential(*list(full_model.children())[:-1], nn.Flatten())
    
    model = model.to(device).eval()
    preprocess = weights.transforms()
    return model, preprocess


def _is_huggingface_model(backbone: str) -> bool:
    """Check if backbone string refers to a HuggingFace model."""
    hf_keywords = ['dinov2', 'dino-', 'vit-', 'beit', 'clip', 'swin', 'convnext']
    
    backbone_lower = backbone.lower()
    
    # Check for HuggingFace-style paths (org/model)
    if '/' in backbone:
        return True
    
    # Check for known HuggingFace model keywords
    for keyword in hf_keywords:
        if keyword in backbone_lower:
            return True
    
    return False


def compute_backbone_embs(
    dataset,
    backbone: str = 'facebook/dinov2-base',
    batch_size: int = 32,
    workers: int = 0,
    device: str = None,
    verbose: bool = True
) -> torch.Tensor:
    """Extract embeddings from a dataset using a backbone model.
    
    Supports both HuggingFace models (DINOv2, ViT, etc.) and torchvision models
    (ResNet, VGG, EfficientNet, etc.). Automatically detects the model type.
    
    Args:
        dataset: Dataset with __getitem__ returning dict with 'inputs'.'x'.
            - For HuggingFace models: expects PIL Images
            - For torchvision models: expects PIL Images or tensors
        backbone (str): Model name for feature extraction. Can be:
            - HuggingFace model: 'facebook/dinov2-base', 'google/vit-base-patch16-224'
            - torchvision model: 'resnet18', 'resnet50', 'vgg16', 'efficientnet_b0'
            Defaults to 'facebook/dinov2-base'.
        batch_size (int, optional): Batch size for processing. Defaults to 32.
        workers (int, optional): Number of DataLoader workers. Defaults to 0.
        device (str, optional): Device to use ('cpu', 'cuda', 'mps', etc.). 
            If None, auto-detects (CUDA > MPS > CPU). Defaults to None.
            Note: MPS may not work with torchvision preprocessing transforms.
        verbose (bool, optional): Print detailed logging information. Defaults to True.
        
    Returns:
        torch.Tensor: Stacked embeddings with shape (n_samples, embedding_dim).
        - DINOv2-base: 768
        - ResNet18: 512
        - ResNet50: 2048
        
    Example:
        >>> # HuggingFace model
        >>> embeddings = compute_backbone_embs(dataset, backbone='facebook/dinov2-base', device='mps')
        >>> embeddings.shape
        torch.Size([10000, 768])
        
        >>> # torchvision model
        >>> embeddings = compute_backbone_embs(dataset, backbone='resnet50', device='cuda')
        >>> embeddings.shape
        torch.Size([10000, 2048])
    """
    
    # Set device with auto-detection if None (priority: CUDA > MPS > CPU)
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    device = torch.device(device)

    if not isinstance(backbone, str):
        raise ValueError("Backbone must be a string representing a model name. "
                        "Use HuggingFace format (e.g., 'facebook/dinov2-base') or "
                        "torchvision format (e.g., 'resnet50'). "
                        "Custom backbones will be supported in future versions.")
    
    # Determine model type and load accordingly
    use_huggingface = _is_huggingface_model(backbone)
    
    if verbose:
        model_type = "HuggingFace" if use_huggingface else "torchvision"
        logger.info(f"Using {model_type} backbone: {backbone}")
        logger.info(f"Device: {device}")
    
    if use_huggingface:
        backbone_model, processor = _load_huggingface_model(backbone, device)
        
        def collate_fn(batch):
            return [sample['inputs']['x'] for sample in batch]
        
        def process_batch(batch_data):
            inputs = processor(images=batch_data, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = backbone_model(**inputs)
            # Use CLS token embedding
            return outputs.last_hidden_state[:, 0, :]
    else:
        backbone_model, preprocess = _load_torchvision_model(backbone, device)
        
        def collate_fn(batch):
            images = [sample['inputs']['x'] for sample in batch]
            # Stack if tensors, otherwise return list for preprocessing
            if isinstance(images[0], torch.Tensor):
                return torch.stack(images)
            return images
        
        def process_batch(batch_data):
            # Handle both tensor and PIL image inputs
            if isinstance(batch_data, list):
                # PIL images - preprocess individually and stack
                from torchvision import transforms
                to_tensor = transforms.ToTensor()
                tensors = [to_tensor(img) for img in batch_data]
                batch_data = torch.stack(tensors)
            batch_data = batch_data.to(device)
            batch_data = preprocess(batch_data)
            return backbone_model(batch_data)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn
    )
    
    # Extract embeddings
    embeddings_list = []
    
    if verbose:
        logger.info("Precomputing embeddings with backbone...")
    
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Extracting embeddings") if verbose else dataloader
        for batch_data in iterator:
            embedding = process_batch(batch_data)
            embeddings_list.append(embedding.cpu())

    all_embeddings = torch.cat(embeddings_list, dim=0)
    
    if verbose:
        logger.info(f"✓ Extracted embeddings with shape: {all_embeddings.shape}")
    
    return all_embeddings


def get_backbone_embs(path: str,
                    dataset,
                    backbone: str = 'facebook/dinov2-base',
                    batch_size: int = 32,
                    force_recompute=False,
                    workers=0,
                    device=None,
                    verbose=True):
    """Get backbone embeddings with automatic caching.
    
    Loads embeddings from cache if available, otherwise computes and saves them.
    This dramatically speeds up training by avoiding repeated (pretrained) backbone computation.
    
    Supports both HuggingFace models (DINOv2, ViT, etc.) and torchvision models
    (ResNet, VGG, EfficientNet, etc.).
    
    Args:
        path (str): File path for saving/loading embeddings (.pt file).
        dataset: Dataset to extract embeddings from.
        backbone (str): Model name for feature extraction. Can be:
            - HuggingFace model: 'facebook/dinov2-base', 'google/vit-base-patch16-224'
            - torchvision model: 'resnet18', 'resnet50', 'vgg16', 'efficientnet_b0'
            Defaults to 'facebook/dinov2-base'.
        batch_size (int): Batch size for computation. Defaults to 32.
        force_recompute (bool, optional): Recompute even if cached. Defaults to False.
        workers (int, optional): Number of DataLoader workers. Defaults to 0.
        device (str, optional): Device to use ('cpu', 'cuda', 'mps', etc.).
            If None, auto-detects (CUDA > MPS > CPU). Defaults to None.
        verbose (bool, optional): Print detailed logging information. Defaults to True.
        
    Returns:
        torch.Tensor: Cached or freshly computed embeddings.
        
    Example:
        >>> # HuggingFace model
        >>> embeddings = get_backbone_embs(
        ...     path='cache/celeba_dinov2.pt',
        ...     dataset=train_dataset,
        ...     backbone='facebook/dinov2-base',
        ...     batch_size=32,
        ...     device='mps'
        ... )
        
        >>> # torchvision model
        >>> embeddings = get_backbone_embs(
        ...     path='cache/celeba_resnet50.pt',
        ...     dataset=train_dataset,
        ...     backbone='resnet50',
        ...     batch_size=64,
        ...     device='cuda'
        ... )
    """
    # if the path of the embeddings are not precomputed and stored, then compute them and store them
    if not os.path.exists(path) or force_recompute:
        # compute
        embs = compute_backbone_embs(dataset,
                                    backbone=backbone,
                                    batch_size=batch_size,
                                    workers=workers,
                                    device=device,
                                    verbose=verbose)
        # save
        if verbose:
            logger.info(f"Saving embeddings to {path}")
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(embs, path)
        if verbose:
            logger.info(f"✓ Saved embeddings with shape: {embs.shape}")

    if verbose:
        logger.info(f"Loading precomputed embeddings from {path}")
    embs = torch.load(path)
    return embs
