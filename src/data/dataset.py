import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch_geometric.loader import LinkNeighborLoader

from src.models.utils import collate_variable


class UniformNegSampler:
    """Uniformly sample negatives from item space (targets)."""

    def __init__(self, num_items, num_neg=1, seed=42):
        self.num_items = num_items
        self.num_neg = num_neg
        self.rng = np.random.default_rng(seed)

    def sample(self, positives=None):
        """Sample negatives excluding given positives if provided."""
        if positives is not None and len(positives) > 0:
            candidates = list(set(range(self.num_items)) - set(positives))
            return self.rng.choice(
                candidates,
                size=self.num_neg,
                replace=len(candidates) < self.num_neg
            )
        return self.rng.integers(low=0, high=self.num_items, size=self.num_neg)


class InteractionDataset(Dataset):
    """
    Dataset for recommendation (NCF or Graph).
    - Train: dynamic resampling of negatives each epoch
    - Val/Test: exhaustive negatives (all items minus positives, static)
    """

    def __init__(self, df, user_map, item_map,
                 num_neg=0, dynamic=False, exhaustive_eval=False, num_eval_negs=None,
                 all_interactions=None, seed=42):
        self.df = df.reset_index(drop=True)
        self.user_map = user_map
        self.item_map = item_map
        self.df["user_idx"] = self.df["user_id"].astype(str).map(self.user_map)
        self.df["item_idx"] = self.df["item_id"].astype(str).map(self.item_map)
        self.num_users = len(user_map)
        self.num_items = len(item_map)

        self.num_neg = num_neg
        self.dynamic = dynamic
        self.exhaustive_eval = exhaustive_eval
        self.num_eval_negs = num_eval_negs  # used only if not exhaustive
        self.rng = np.random.default_rng(seed)

        self.all_interactions = all_interactions if all_interactions else {}
        self.neg_items = {
            u: np.setdiff1d(np.arange(self.num_items), list(pos))
            for u, pos in self.all_interactions.items()
}
        # This gets all the users that have positive interaction before temporal split
        self.users_in_df = set(self.user_map[str(u)] for u in self.df["user_id"].astype(str))
        self.sampler = UniformNegSampler(self.num_items, num_neg, seed) if num_neg > 0 else None

        # expanded samples (u, i, label)
        self.samples = []

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
        """Expand positives + fresh negatives for training (only users with positives in df)."""
        samples = []
        for _, row in self.df.iterrows():
            u, i, score = row["user_idx"], row["item_idx"], float(row["label"])
            if u not in self.users_in_df:
                continue
            samples.append((u, i, score))  # use score from df

            positives = self.all_interactions.get(u, set())
            if self.sampler:
                negs = self.sampler.sample(positives)
                for n in negs:
                    samples.append((u, n, 0.0))  # negatives are always 0
        self.samples = samples
    # -----------------------
    # Eval negatives (static exhaustive)
    # -----------------------
    def _build_exhaustive_samples(self):
        """Positives + all other items as negatives (per user)."""
        samples = []
        all_items = set(self.item_map.values())

        for u in self.users_in_df:
            positives = self.all_interactions.get(u, set())
            user_rows = self.df[self.df["user_idx"] == u]

            # add positives with score
            for _, row in user_rows.iterrows():
                samples.append((u, row["item_idx"], float(row["label"])))

            # add all remaining as negatives
            for n in all_items - positives:
                samples.append((u, n, 0.0))

        self.samples = samples

    # -----------------------
    # Eval negatives (sampled)
    # -----------------------
    def _build_sampled_eval_samples(self):
        """Positives (with scores) + K sampled negatives per user."""
        samples = []
        all_items = np.array(list(self.item_map.values()))

        for u in self.users_in_df:
            positives = self.all_interactions.get(u, set())
            user_rows = self.df[self.df["user_idx"] == u]

            # add positives with score
            for _, row in user_rows.iterrows():
                samples.append((u, row["item_idx"], float(row["label"])))

            # sample negatives
            candidates = np.setdiff1d(all_items, list(positives))
            negs = self.rng.choice(
                candidates,
                size=min(self.num_eval_negs, len(candidates)),
                replace=False
            )
            for n in negs:
                samples.append((u, n, 0.0))

        self.samples = samples

    # -----------------------
    # Dataset API
    # -----------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, i, l = self.samples[idx]
        return {
            "user_id": torch.tensor(u),
            "item_id": torch.tensor(i),
            "label": torch.tensor(l),
        }

    # -----------------------
    # Loader Builders
    # -----------------------
    def build_ncf_loader(self, batch_size=512, shuffle=True):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_variable)

    def build_graph_loader(self, hetero_graph, batch_size=1024, num_neighbors=[15, 10], shuffle=True):
        edge_type = ("diseases", "clinical_trial", "targets")
        users, items, labels = zip(*self.samples)
        edge_label_index = torch.tensor([users, items], dtype=torch.long)
        edge_label = torch.tensor(labels, dtype=torch.float)

        return LinkNeighborLoader(
            data=hetero_graph,
            edge_label_index=(edge_type, edge_label_index),
            edge_label=edge_label,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
        )

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
