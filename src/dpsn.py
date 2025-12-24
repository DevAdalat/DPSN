import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple, List, Dict


class ParameterPool(nn.Module):
    def __init__(self, pool_size: int = 1_000_000, param_dim: int = 768):
        super().__init__()
        self.pool_size = pool_size
        self.param_dim = param_dim
        self.params = nn.Parameter(torch.randn(pool_size, param_dim) * 0.02)


class RouteGeneratorModel(nn.Module):
    def __init__(self, input_dim: int, pool_size: int):
        super().__init__()
        self.pool_size = pool_size
        self.hidden_dim = 256

        self.complexity_net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

        self.scorer = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, pool_size),
        )

    def determine_budget_batch(self, complexities: torch.Tensor) -> int:
        """
        complexities: [Batch, 1]
        Returns a single budget integer to use for this batch (Max or Mean strategy).
        For simplicity and tensor regularity, we use the MAX complexity in the batch
        to ensure the 'hardest' sample gets enough parameters.
        """
        min_params = 100
        max_params = 5000

        # Use mean complexity of the batch to decide the budget for the batch
        # This keeps the tensor uniform [Batch, Budget, Dim]
        avg_complexity = complexities.mean().item()

        scale = avg_complexity**2
        budget = int(min_params + (max_params - min_params) * scale)
        return budget

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        # x: [Batch, InputDim]

        # 1. Complexity
        complexity_scores = self.complexity_net(x)  # [Batch, 1]
        budget = self.determine_budget_batch(complexity_scores)

        # 2. Scores
        scores = self.scorer(x)  # [Batch, PoolSize]

        # 3. TopK
        if self.training:
            noise = torch.randn_like(scores) * 0.1
            scores_with_noise = scores + noise
            top_scores, indices = torch.topk(scores_with_noise, k=budget, dim=1)
        else:
            top_scores, indices = torch.topk(scores, k=budget, dim=1)

        weights = torch.softmax(top_scores, dim=1)

        return indices, budget, weights, complexity_scores


class SparseExecutionEngine(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(
        self,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        pool: ParameterPool,
    ) -> torch.Tensor:
        # x: [Batch, InputDim]
        # indices: [Batch, Budget]

        # 1. Fetch params
        # pool.params[indices] handles the batch gathering
        # result: [Batch, Budget, ParamDim]
        selected_params = pool.params[indices]

        # 2. Matrix Mult
        # x.unsqueeze(1): [Batch, 1, InputDim]
        # selected_params.transpose(1, 2): [Batch, ParamDim, Budget]
        # output: [Batch, 1, Budget]
        products = torch.matmul(x.unsqueeze(1), selected_params.transpose(1, 2))

        # 3. Activation
        activations = torch.tanh(products)
        weighted_activations = activations * weights.unsqueeze(1)

        # 4. Reconstruct
        # [Batch, 1, Budget] @ [Batch, Budget, ParamDim] -> [Batch, 1, ParamDim]
        output = torch.matmul(weighted_activations, selected_params)

        return x + output.squeeze(1)


class DPSN(nn.Module):
    def __init__(self, input_dim: int = 768, pool_size: int = 100_000):
        super().__init__()
        self.pool = ParameterPool(pool_size, input_dim)
        self.router = RouteGeneratorModel(input_dim, pool_size)
        self.executor = SparseExecutionEngine(input_dim)

    def forward(self, x: torch.Tensor) -> Dict:
        # x: [Batch, InputDim]
        indices, budget, weights, complexity = self.router(x)

        start_time = time.time()
        output = self.executor(x, indices, weights, self.pool)
        end_time = time.time()

        return {
            "output": output,
            "complexity_score": complexity,  # [Batch, 1]
            "parameters_used": budget,  # int
            "indices": indices,
            "execution_time_ms": (end_time - start_time) * 1000,
        }
