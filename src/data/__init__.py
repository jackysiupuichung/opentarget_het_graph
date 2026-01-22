"""Data module for heterogeneous graph benchmarking."""

from .utils import temporal_split, cold_start_split, attach_node_features

__all__ = [
    "temporal_split",
    "cold_start_split",
    "attach_node_features",
]
