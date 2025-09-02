import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.nn.utils.rnn import pad_sequence


from src.models.ncf import NCF
from src.models.hetgatv2 import HetGATv2

# from src.models.temporal_hetgat import TemporalHetGAT
# from src.models.temporal_hettfm import TemporalHetTransformer


def _infer_node_dims(hetero_data):
    """Infer input dims and num_nodes for all node types in HeteroData."""
    node_in_dims = {}
    num_nodes = {}
    for nt in hetero_data.node_types:
        num_nodes[nt] = hetero_data[nt].num_nodes
        x = getattr(hetero_data[nt], "x", None)
        node_in_dims[nt] = int(x.size(-1)) if x is not None else 0
    return node_in_dims, num_nodes


def initialise_model(cfg, user_map, item_map, hetero_data=None, pretrained_embeddings=None):
    """
    Initialise the recommender model.

    Args:
        cfg: config object (YAML via OmegaConf/Hydra)
        user_map: dict {user_id -> index}, built from ALL disease nodes
        item_map: dict {item_id -> index}, built from ALL target nodes
        hetero_data: PyG HeteroData (for graph models)
        pretrained_embeddings: optional dict with pretrained embeddings per node type

    Returns:
        model (torch.nn.Module)
    """

    model_name = cfg.model.name.lower()

    # --------------------
    # Classic Neural CF
    # --------------------
    if model_name == "ncf":
        num_users = len(user_map)
        num_items = len(item_map)

        return NCF(
            num_users=num_users,
            num_items=num_items,
            embed_dim=cfg.model.embed_dim,
            user_emb=pretrained_embeddings.get("user") if pretrained_embeddings else None,
            item_emb=pretrained_embeddings.get("item") if pretrained_embeddings else None,
        )

    # --------------------
    # Graph-based: HetGATv2
    # --------------------
    elif model_name == "gat":
        if hetero_data is None:
            raise ValueError("Graph model requires hetero_data")

        metadata = hetero_data.metadata()
        _, num_nodes = _infer_node_dims(hetero_data)

        return HetGATv2(
            metadata=metadata,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            heads=cfg.model.heads,
            num_nodes=num_nodes,
            embedding_dim=getattr(cfg.model, "embedding_dim", cfg.model.hidden_dim),
            pretrained_embeddings=pretrained_embeddings,
            pair_src_type=cfg.model.supervision_src_type,
            pair_dst_type=cfg.model.supervision_dst_type,
            pair_mlp_hidden=cfg.model.mlp_hidden,
            dropout=cfg.model.dropout,
        )

    # --------------------
    # Temporal HetGAT
    # --------------------
    elif model_name == "t-hgat":
        raise NotImplementedError("TemporalHetGAT integration not yet implemented")

    # --------------------
    # Temporal HetTransformer
    # --------------------
    elif model_name == "t-hgt":
        raise NotImplementedError("TemporalHetTransformer integration not yet implemented")

    else:
        raise ValueError(f"❌ Unknown model: {cfg.model.name}")


def initialise_trainer(cfg, run_dir):
    """
    Initialise PyTorch Lightning Trainer with callbacks and monitoring.

    Args:
        cfg: config object
        run_dir: experiment directory

    Returns:
        trainer, checkpoint_callback
    """

    # -----------------------
    # Dynamic monitor metric
    # -----------------------
    if cfg.model.loss_type in ["mse", "bce"]:
        monitor_metric, mode = "val_loss", "min"
    else:
        monitor_metric, mode = f"val_{cfg.eval.valid_metric}", "max"

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_model",
        save_top_k=1,
        monitor=monitor_metric,
        mode=mode,
    )

    earlystop_cb = EarlyStopping(
        monitor=monitor_metric,
        patience=getattr(cfg.train, "patience", 20),
        mode=mode
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        devices=1,
        default_root_dir=run_dir,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, earlystop_cb],
    )

    return trainer, checkpoint_cb


def collate_variable(batch):
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


