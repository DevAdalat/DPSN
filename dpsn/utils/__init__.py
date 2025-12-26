"""Utility functions for DPSN."""

import torch
from typing import Dict, List


def calculate_sparsity(grad: torch.Tensor) -> float:
    if grad.is_sparse:
        grad_dense = grad.to_dense()
    else:
        grad_dense = grad

    total_elements = grad_dense.numel()
    nonzero_elements = (grad_dense != 0).sum().item()

    return 1.0 - (nonzero_elements / total_elements)


def count_unique_parameters_used(indices_list: List[torch.Tensor]) -> int:
    all_indices = set()
    for indices in indices_list:
        all_indices.update(indices.flatten().tolist())
    return len(all_indices)


def get_parameter_overlap(indices1: torch.Tensor, indices2: torch.Tensor) -> float:
    set1 = set(indices1.flatten().tolist())
    set2 = set(indices2.flatten().tolist())

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def format_number(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)
