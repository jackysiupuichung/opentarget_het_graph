import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch_geometric.loader import LinkNeighborLoader, HGTLoader

from src.models.utils import collate_variable


class UniformNegSampler:
    """Uniformly sample negatives from precomputed candidate pools."""

    def __init__(self, num_neg=1, seed=42):
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)

    def sample(self, candidates):
        if len(candidates) == 0:
            return np.array([], dtype=int)
        return self.rng.choice(
            candidates,
            size=min(self.num_neg, len(candidates)),
            replace=False if len(candidates) >= self.num_neg else True
        )

class InteractionDataset(Dataset):
    """
    Dataset for recommendation (NCF or Graph).
    - Train: dynamic resampling of negatives each epoch
    - Val/Test: exhaustive negatives or sampled negatives per user
    """

    def __init__(self, df, user_map, item_map,
                 num_neg=0, dynamic=False,
                 exhaustive_eval=False, num_eval_negs=None,
                 all_interactions=None, seed=42):
        self.df = df.reset_index(drop=True).copy()
        self.user_map = user_map
        self.item_map = item_map

        # map ids once
        self.df["user_idx"] = self.df["user_id"].astype(str).map(self.user_map)
        self.df["item_idx"] = self.df["item_id"].astype(str).map(self.item_map)

        self.num_users = len(user_map)
        self.num_items = len(item_map)

        self.num_neg = num_neg
        self.dynamic = dynamic
        self.exhaustive_eval = exhaustive_eval
        self.num_eval_negs = num_eval_negs
        self.rng = np.random.default_rng(seed)

        self.all_interactions = all_interactions if all_interactions else {}
        self.neg_items = {
            u: np.setdiff1d(np.arange(self.num_items), list(pos))
            for u, pos in self.all_interactions.items()
        }

        # only keep users with positives in this df
        self.users_in_df = set(self.df["user_idx"].unique())

        self.sampler = UniformNegSampler(num_neg, seed) if num_neg > 0 else None

        # expanded samples (N, 3) array
        self.samples = np.empty((0, 3))

        if self.exhaustive_eval:
            self._build_exhaustive_samples()
        elif self.num_eval_negs is not None:
            self._build_sampled_eval_samples()
        else:
            self.resample()

        print("✅ InteractionDataset built:", self.dataset_description())

    # -----------------------
    # Train negatives (dynamic)
    # -----------------------
    def resample(self):
        # positives
        user_idx = self.df["user_idx"].to_numpy()
        item_idx = self.df["item_idx"].to_numpy()
        labels = self.df["label"].astype(float).to_numpy()
        pos_samples = np.stack([user_idx, item_idx, labels], axis=1)

        # negatives
        neg_samples = []
        if self.sampler:
            for u in self.users_in_df:
                candidates = self.neg_items.get(u, np.arange(self.num_items))
                negs = self.sampler.sample(candidates)
                if len(negs) > 0:
                    users = np.full(len(negs), u, dtype=int)
                    labels = np.zeros(len(negs), dtype=float)
                    neg_samples.append(np.stack([users, negs, labels], axis=1))
        neg_samples = np.concatenate(neg_samples, axis=0) if neg_samples else np.empty((0, 3))

        self.samples = np.concatenate([pos_samples, neg_samples], axis=0)

    # -----------------------
    # Eval negatives (static exhaustive)
    # -----------------------
    def _build_exhaustive_samples(self):
        # positives
        user_idx = self.df["user_idx"].to_numpy()
        item_idx = self.df["item_idx"].to_numpy()
        labels = self.df["label"].astype(float).to_numpy()
        pos_samples = np.stack([user_idx, item_idx, labels], axis=1)

        # negatives = all items except positives
        neg_samples = []
        for u in self.users_in_df:
            candidates = self.neg_items.get(u, [])
            if len(candidates) > 0:
                users = np.full(len(candidates), u, dtype=int)
                labels = np.zeros(len(candidates), dtype=float)
                neg_samples.append(np.stack([users, candidates, labels], axis=1))
        neg_samples = np.concatenate(neg_samples, axis=0) if neg_samples else np.empty((0, 3))

        self.samples = np.concatenate([pos_samples, neg_samples], axis=0)

    # -----------------------
    # Eval negatives (sampled)
    # -----------------------
    def _build_sampled_eval_samples(self):
        # positives
        user_idx = self.df["user_idx"].to_numpy()
        item_idx = self.df["item_idx"].to_numpy()
        labels = self.df["label"].astype(float).to_numpy()
        pos_samples = np.stack([user_idx, item_idx, labels], axis=1)

        # negatives
        neg_samples = []
        for u in self.users_in_df:
            candidates = self.neg_items.get(u, [])
            if len(candidates) > 0:
                negs = self.rng.choice(
                    candidates,
                    size=min(self.num_eval_negs, len(candidates)),
                    replace=False
                )
                users = np.full(len(negs), u, dtype=int)
                labels = np.zeros(len(negs), dtype=float)
                neg_samples.append(np.stack([users, negs, labels], axis=1))
        neg_samples = np.concatenate(neg_samples, axis=0) if neg_samples else np.empty((0, 3))

        self.samples = np.concatenate([pos_samples, neg_samples], axis=0)

    # -----------------------
    # Dataset API
    # -----------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, i, l = self.samples[idx]
        return {
            "user_id": torch.tensor(int(u)),
            "item_id": torch.tensor(int(i)),
            "label": torch.tensor(float(l)),
        }

    # -----------------------
    # Loader Builders
    # -----------------------
    def build_ncf_loader(self, batch_size=512, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_variable)

    def build_graph_loader(
        self,
        hetero_graph,
        batch_size=1024,
        num_neighbors=[15, 10],
        shuffle=True,
    ):
        edge_type = ("diseases", "clinical_trial", "targets")
        assert edge_type in hetero_graph.edge_types, f"{edge_type} missing in graph {hetero_graph.edge_types}"

        users = torch.as_tensor(self.samples[:, 0], dtype=torch.long)
        items = torch.as_tensor(self.samples[:, 1], dtype=torch.long)
        labels = torch.as_tensor(self.samples[:, 2], dtype=torch.float)

        # shape [2, N]
        edge_label_index = torch.stack([users, items], dim=0)
        assert edge_label_index.shape[0] == 2, f"edge_label_index wrong shape {edge_label_index.shape}"
        assert edge_label_index.shape[1] == labels.shape[0], "edge_label_index and labels mismatch"

        return LinkNeighborLoader(
            data=hetero_graph,
            edge_label_index=(edge_type, edge_label_index),
            edge_label=labels,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        
    def build_HGT_loader(
        self,
        hetero_graph,
        batch_size=1024,
        num_neighbors=[15, 10],
        shuffle=True
    ):

        """
        Build graph loader using HGTLoader instead of LinkNeighborLoader.
        This ensures schema-aware, type-balanced sampling.
        """

        edge_type = ("diseases", "clinical_trial", "targets")
        assert edge_type in hetero_graph.edge_types, \
            f"{edge_type} missing in graph {hetero_graph.edge_types}"

        users = torch.as_tensor(self.samples[:, 0], dtype=torch.long)
        items = torch.as_tensor(self.samples[:, 1], dtype=torch.long)
        labels = torch.as_tensor(self.samples[:, 2], dtype=torch.float)

        # edge_label_index for supervision
        edge_label_index = torch.stack([users, items], dim=0)

        loader = HGTLoader(
            data=hetero_graph,
            num_samples={key: [num_neighbors] * 2 for key in hetero_graph.edge_types},
            # supervision edges
            input_nodes=(edge_type[0], users),  # starting from diseases
            batch_size=batch_size,
            shuffle=shuffle,
        )

        return loader

    # -----------------------
    # Info
    # -----------------------
    def dataset_description(self):
        return {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "num_pos_interactions": len(self.df),
            "num_pos_users": len(self.users_in_df),
            "num_samples": len(self.samples),
            "num_neg_per_pos": self.num_neg,
            "dynamic_neg_sampling": self.dynamic,
            "exhaustive_eval": self.exhaustive_eval,
        }
