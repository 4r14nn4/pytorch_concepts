import torch
from torch import nn
from typing import List, Optional, Mapping, Union

from .....data.utils import ensure_list
from .....annotations import Annotations

from ...low.dense_layers import MLP
from ..base.model import BaseModel
from ..learners import JointLearner



class BlackBox(BaseModel, JointLearner):
    """
    BlackBox model.

    This model implements a standard neural network architecture for concept-based tasks,
    without explicit concept bottleneck or interpretable intermediate representations.
    It uses a backbone for feature extraction and a latent encoder for concepts prediction.

    Args:
        input_size (int): Dimensionality of input features.
        annotations (Annotations): Annotation object for output variables.
        loss (nn.Module, optional): Loss function for training.
        metrics (Mapping, optional): Metrics for evaluation.
        backbone (nn.Module, optional): Feature extraction module.
        latent_encoder (nn.Module, optional): Latent encoder module.
        latent_encoder_kwargs (dict, optional): Arguments for latent encoder.
        **kwargs: Additional arguments for BaseModel.

    Example:
        >>> model = BlackBox(input_size=8, annotations=ann)
        >>> out = model(torch.randn(2, 8))
    """
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        variable_distributions: Optional[Mapping] = None,
        loss: Optional[nn.Module] = None,
        metrics: Optional[Mapping] = None,
        inference: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            variable_distributions=variable_distributions,
            loss=loss,
            metrics=metrics,
            **kwargs
        )
        output_size = sum(self.concept_annotations.cardinalities)
        self.linear = nn.Linear(self.latent_size, output_size)

    def forward(self,
                x: torch.Tensor,
                query: List[str] = None,
        ) -> torch.Tensor:
        features = self.maybe_apply_backbone(x)
        endogenous = self.latent_encoder(features)
        output = self.linear(endogenous)
        return output

    def filter_output_for_loss(self, forward_out, target):
        """No filtering needed - return raw endogenous for standard loss computation.

        Args:
            forward_out: Model output endogenous.
            target: Ground truth labels.

        Returns:
            Dict with 'input' and 'target' for loss computation.
        """
        # forward_out: endogenous
        # return: endogenous
        return {'input': forward_out,
                'target': target}

    def filter_output_for_metrics(self, forward_out, target):
        """No filtering needed - return raw endogenous for metric computation.

        Args:
            forward_out: Model output endogenous.
            target: Ground truth labels.

        Returns:
            Dict with 'input' and 'target' for metric computation.
        """
        # forward_out: endogenous
        # return: endogenous
        return {'preds': forward_out,
                'target': target}
    


class BlackBoxTaskOnly(BaseModel, JointLearner):
    """
    BlackBox model.

    This model implements a standard neural network architecture for predicting tasks only,
    without explicit concept bottleneck or interpretable intermediate representations.
    It uses a backbone for feature extraction and a latent encoder for concepts prediction.

    Args:
        input_size (int): Dimensionality of input features.
        annotations (Annotations): Annotation object for output variables.
        variable_distributions (Mapping, optional): Distributions of variables.
        loss (nn.Module, optional): Loss function for training.
        metrics (Mapping, optional): Metrics for evaluation.
        backbone (nn.Module, optional): Feature extraction module.
        latent_encoder (nn.Module, optional): Latent encoder module.
        latent_encoder_kwargs (dict, optional): Arguments for latent encoder.
        **kwargs: Additional arguments for BaseModel.

    Example:
        >>> model = BlackBox(input_size=8, annotations=ann)
        >>> out = model(torch.randn(2, 8))
    """
    def __init__(
        self,
        input_size: int,
        annotations: Annotations,
        task_names: Union[List[str], str],
        variable_distributions: Optional[Mapping] = None,
        loss: Optional[nn.Module] = None,
        metrics: Optional[Mapping] = None,
        inference: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            input_size=input_size,
            annotations=annotations,
            variable_distributions=variable_distributions,
            loss=loss,
            metrics=metrics,
            **kwargs
        )
        # extract only task output size
        task_names = ensure_list(task_names)
        
        # Extract concept cardinalities (excluding tasks)
        task_idxs = [self.concept_names.index(name) for name in task_names]
        task_cardinalities = [self.concept_annotations.cardinalities[i] for i in task_idxs]
        output_size = sum(task_cardinalities)
        # also compute total cardinality
        self.total_cardinality = sum(self.concept_annotations.cardinalities)
        # Compute column indices for task placement
        self.task_start_idx = sum(self.concept_annotations.cardinalities[:task_idxs[0]])
        self.task_end_idx = self.task_start_idx + output_size
        self.linear = nn.Linear(self.latent_size, output_size)

    def forward(self,
                x: torch.Tensor,
                query: List[str] = None,
        ) -> torch.Tensor:
        features = self.maybe_apply_backbone(x)
        endogenous = self.latent_encoder(features)
        output = self.linear(endogenous)
        return output

    def filter_output_for_loss(self, forward_out, target):
        """Pad predictions with zeros to match total cardinality.

        Args:
            forward_out: Model output (task predictions only).
            target: Ground truth labels.

        Returns:
            Dict with 'input' (padded predictions) and 'target' for loss computation.
        """
        # Create padded output with zeros
        batch_size = forward_out.shape[0]
        padded_output = torch.zeros((batch_size, self.total_cardinality),
                                    dtype=forward_out.dtype, device=forward_out.device)
        # Fill in task predictions at correct positions
        padded_output[:, self.task_start_idx:self.task_end_idx] = forward_out
        return {'input': padded_output,
                'target': target}

    def filter_output_for_metrics(self, forward_out, target):
        """Pad predictions with zeros to match total cardinality.

        Args:
            forward_out: Model output (task predictions only).
            target: Ground truth labels.

        Returns:
            Dict with 'preds' (padded predictions) and 'target' for metric computation.
        """
        # Create padded output with zeros
        batch_size = forward_out.shape[0]
        padded_output = torch.zeros((batch_size, self.total_cardinality),
                                    dtype=forward_out.dtype, device=forward_out.device)
        # Fill in task predictions at correct positions
        padded_output[:, self.task_start_idx:self.task_end_idx] = forward_out
        return {'preds': padded_output,
                'target': target}