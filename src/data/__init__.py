"""Data module for heterogeneous graph benchmarking."""

import wandb
from omegaconf import OmegaConf, DictConfig
from .utils import temporal_split, cold_start_split, attach_node_features
from .temporal_loader import to_temporal_snapshots

def init_wandb(cfg: DictConfig):
    """
    Initialize WandB if enabled in config.
    """
    if cfg.get("wandb", {}).get("enabled", False):
        wandb.init(
            project=cfg.wandb.get("project", "opentargets-graph"),
            entity=cfg.wandb.get("entity", None),
            name=cfg.wandb.get("name", None) or cfg.get("experiment_name", "default"),
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True
        )
        print(f"🚀 WandB initialized: {wandb.run.name}")
    else:
        print("🚫 WandB disabled")

__all__ = [
    "temporal_split",
    "cold_start_split",
    "attach_node_features",
    "to_temporal_snapshots",
    "init_wandb",
]
