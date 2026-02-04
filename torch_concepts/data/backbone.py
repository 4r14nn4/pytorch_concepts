"""Backbone utilities for feature extraction and embedding precomputation.

Provides functions to extract and cache embeddings from pre-trained backbone
models (e.g., ResNet, ViT) to speed up training of concept-based models.
"""
import os
import torch
import logging
from torch.utils.data import DataLoader
from torchvision.models import get_model, get_model_weights
from tqdm import tqdm


logger = logging.getLogger(__name__)

def compute_backbone_embs(
    dataset,
    backbone: str,
    batch_size: int = 512,
    workers: int = 0,
    device: str = None,
    verbose: bool = True
) -> torch.Tensor:
    """Extract embeddings from a dataset using a backbone model.
    
    Performs a forward pass through the backbone for the entire dataset and
    returns the concatenated embeddings. Useful for precomputing features
    to avoid repeated backbone computation during training.
    
    Args:
        dataset: Dataset with __getitem__ returning dict with 'x' key or 'inputs'.'x' nested key.
        backbone (str): Backbone model name for feature extraction (e.g., 'resnet18').
        batch_size (int, optional): Batch size for processing. Defaults to 512.
        workers (int, optional): Number of DataLoader workers. Defaults to 0.
        device (str, optional): Device to use ('cpu', 'cuda', 'cuda:0', etc.). 
            If None, auto-detects ('cuda' if available, else 'cpu'). Defaults to None.
        verbose (bool, optional): Print detailed logging information. Defaults to True.
        
    Returns:
        torch.Tensor: Stacked embeddings with shape (n_samples, embedding_dim).
        
    Example:
        >>> from torchvision.models import resnet18
        >>> backbone = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        >>> embeddings = compute_backbone_embs(my_dataset, backbone, batch_size=64, device='cuda')
        >>> embeddings.shape
        torch.Size([10000, 512])
    """
    
    # Set device with auto-detection if None (priority: CUDA > MPS > CPU)
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        # elif torch.backends.mps.is_available():
            # device = 'mps' # NOT SUPPORTED
        else:
            device = 'cpu'
    device = torch.device(device)
    
    if isinstance(backbone, str):
        # Get weights enum and access DEFAULT to get actual weights instance
        weights = get_model_weights(backbone).DEFAULT
        backbone_model = get_model(backbone, weights=weights).to(device).eval()
        preprocess = weights.transforms()
    else:
        raise ValueError("Backbone must be a string model name \
            supported by torchvision.models. Custom backbones will \
            be supported soon.")
    
    # Custom collate function to extract 'x' from nested dict structure
    def collate_fn(batch):
        return torch.stack([sample['inputs']['x'] for sample in batch])
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn
    )
    
    embeddings_list = []
    
    if verbose:
        logger.info("Precomputing embeddings with backbone...")
    with torch.no_grad():
        iterator = tqdm(dataloader, desc="Extracting embeddings") if verbose else dataloader
        for batch in iterator:
            batch = batch.to(device)
            batch = preprocess(batch)  # Apply preprocessing transforms
            embedding = backbone_model(batch)  # Forward pass through backbone
            embeddings_list.append(embedding.cpu())  # Move back to CPU and store

    all_embeddings = torch.cat(embeddings_list, dim=0)  # Concatenate all embeddings
    
    return all_embeddings

def get_backbone_embs(path: str,
                    dataset,
                    backbone: str,
                    batch_size,
                    force_recompute=False,
                    workers=0,
                    device=None,
                    verbose=True):
    """Get backbone embeddings with automatic caching.
    
    Loads embeddings from cache if available, otherwise computes and saves them.
    This dramatically speeds up training by avoiding repeated (pretrained) backbone computation.
    
    Args:
        path (str): File path for saving/loading embeddings (.pt file).
        dataset: Dataset to extract embeddings from.
        backbone: Backbone model name for feature extraction.
        batch_size: Batch size for computation.
        force_recompute (bool, optional): Recompute even if cached. Defaults to False.
        workers (int, optional): Number of DataLoader workers. Defaults to 0.
        device (str, optional): Device to use ('cpu', 'cuda', 'cuda:0', etc.).
            If None, auto-detects ('cuda' if available, else 'cpu'). Defaults to None.
        verbose (bool, optional): Print detailed logging information. Defaults to True.
        
    Returns:
        torch.Tensor: Cached or freshly computed embeddings.
        
    Example:
        >>> embeddings = get_backbone_embs(
        ...     path='cache/mnist_resnet18.pt',
        ...     dataset=train_dataset,
        ...     backbone=my_backbone,
        ...     batch_size=256,
        ...     device='cuda'
        ... )
        Loading precomputed embeddings from cache/mnist_resnet18.pt
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
