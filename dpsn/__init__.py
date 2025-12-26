"""Dynamic Parameter Selection Network (DPSN) - A modular brain architecture."""

from dpsn.core.parameter_pool import ParameterPool
from dpsn.core.router import RouteGeneratorModel
from dpsn.core.executor import SparseExecutionEngine
from dpsn.core.dpsn import DPSN
from dpsn.models.language_model import DPSNLanguageModel

__version__ = "1.0.0"

__all__ = [
    "ParameterPool",
    "RouteGeneratorModel",
    "SparseExecutionEngine",
    "DPSN",
    "DPSNLanguageModel",
]
