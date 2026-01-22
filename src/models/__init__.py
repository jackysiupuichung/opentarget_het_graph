"""Model architectures for heterogeneous graph benchmarking."""

from .hgt import HGT, HGTLinkPredictor
from .utils import build_model, get_metadata

__all__ = [
    "HGT",
    "HGTLinkPredictor",
    "build_model",
    "get_metadata",
]
