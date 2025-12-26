"""Route Generator Model - The intelligent decision-maker."""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RouteGeneratorModel(nn.Module):
    """Dynamically routes inputs to appropriate parameters based on complexity.

    The router is the "librarian" of the DPSN architecture. It analyzes each
    input to determine:
    1. Complexity: How difficult is this input? (determines budget)
    2. Routing: Which specific parameters should be activated?

    This enables adaptive computation where simple inputs use fewer parameters
    and complex inputs use more, similar to human cognitive resource allocation.

    Args:
        input_dim: Dimension of input vectors
        pool_size: Total number of parameters available in the pool
        hidden_dim: Hidden dimension for scoring network
        min_params: Minimum number of parameters to select
        max_params: Maximum number of parameters to select
        complexity_exponent: Exponent for complexity scaling (default: 2)
        noise_std: Standard deviation of exploration noise during training

    Attributes:
        pool_size: Size of the parameter pool
        hidden_dim: Hidden dimension of scorer
        min_params: Minimum budget
        max_params: Maximum budget
        complexity_exponent: Scaling factor for budget calculation
        noise_std: Exploration noise magnitude
        complexity_net: Network that estimates input complexity
        scorer: Network that scores all parameters for relevance
    """

    def __init__(
        self,
        input_dim: int,
        pool_size: int,
        hidden_dim: int = 256,
        min_params: int = 100,
        max_params: int = 5000,
        complexity_exponent: float = 2.0,
        noise_std: float = 0.1,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.hidden_dim = hidden_dim
        self.min_params = min_params
        self.max_params = max_params
        self.complexity_exponent = complexity_exponent
        self.noise_std = noise_std

        # Complexity estimation network
        # Maps input -> scalar complexity score in [0, 1]
        self.complexity_net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

        # Parameter scoring network
        # Maps input -> relevance score for each parameter in pool
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pool_size),
        )

    def determine_budget(self, complexity_scores: torch.Tensor) -> int:
        """Determine parameter budget based on complexity scores.

        Uses the mean complexity across the batch to determine a uniform
        budget for all samples in the batch. This ensures tensor regularity
        while still adapting to input difficulty.

        Budget scales non-linearly with complexity using the formula:
        budget = min_params + (max_params - min_params) * complexity^exponent

        Args:
            complexity_scores: Tensor of shape [batch_size, 1] with complexity
                             scores in range [0, 1]

        Returns:
            Integer budget (number of parameters to select)
        """
        avg_complexity = complexity_scores.mean().item()

        scale = avg_complexity**self.complexity_exponent

        budget = int(self.min_params + (self.max_params - self.min_params) * scale)

        budget = max(self.min_params, min(self.max_params, min(self.pool_size, budget)))

        return budget

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        """Select parameters based on input complexity and relevance.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Tuple containing:
                - indices: Selected parameter indices [batch_size, budget]
                - budget: Number of parameters selected (int)
                - weights: Softmax weights for selected params [batch_size, budget]
                - complexity_scores: Complexity estimates [batch_size, 1]
        """
        # Step 1: Estimate complexity
        complexity_scores = self.complexity_net(x)  # [batch_size, 1]
        budget = self.determine_budget(complexity_scores)

        # Step 2: Score all parameters
        scores = self.scorer(x)  # [batch_size, pool_size]

        # Step 3: Select top-k with exploration noise during training
        if self.training and self.noise_std > 0:
            # Add Gaussian noise for exploration
            noise = torch.randn_like(scores) * self.noise_std
            scores_with_noise = scores + noise
            top_scores, indices = torch.topk(scores_with_noise, k=budget, dim=1)
        else:
            # Deterministic selection during evaluation
            top_scores, indices = torch.topk(scores, k=budget, dim=1)

        # Step 4: Compute softmax weights for selected parameters
        weights = torch.softmax(top_scores, dim=1)

        return indices, budget, weights, complexity_scores

    def get_routing_statistics(self, x: torch.Tensor) -> dict:
        """Get detailed statistics about routing decisions.

        Useful for analysis and debugging.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Dictionary containing routing statistics
        """
        with torch.no_grad():
            complexity_scores = self.complexity_net(x)
            scores = self.scorer(x)
            budget = self.determine_budget(complexity_scores)

            return {
                "mean_complexity": complexity_scores.mean().item(),
                "std_complexity": complexity_scores.std().item(),
                "min_complexity": complexity_scores.min().item(),
                "max_complexity": complexity_scores.max().item(),
                "budget": budget,
                "budget_ratio": budget / self.pool_size,
                "mean_score": scores.mean().item(),
                "std_score": scores.std().item(),
            }
