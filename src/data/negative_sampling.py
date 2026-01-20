#!/usr/bin/env python3
"""
Negative sampling strategies for link prediction.
"""

import torch
import numpy as np
from typing import Set, List, Dict, Optional


class NegativeSampler:
    """
    Negative sampler for link prediction tasks.
    
    Supports random negative sampling with filtering to avoid existing edges.
    """
    
    def __init__(
        self,
        num_items: int,
        all_interactions: Optional[Dict[int, Set[int]]] = None,
        seed: int = 42,
    ):
        """
        Initialize negative sampler.
        
        Args:
            num_items: Total number of items (targets)
            all_interactions: Dict mapping user ID to set of interacted item indices
            seed: Random seed
        """
        self.num_items = num_items
        self.all_interactions = all_interactions or {}
        self.rng = np.random.RandomState(seed)
    
    def sample(
        self,
        user_idx: int,
        num_negatives: int = 1,
        exclude_items: Optional[Set[int]] = None,
    ) -> List[int]:
        """
        Sample negative items for a user.
        
        Args:
            user_idx: User index
            num_negatives: Number of negative samples
            exclude_items: Additional items to exclude
            
        Returns:
            List of negative item indices
        """
        # Get items to exclude (positive interactions)
        exclude = self.all_interactions.get(user_idx, set())
        if exclude_items is not None:
            exclude = exclude | exclude_items
        
        # Sample negatives
        negatives = []
        max_attempts = num_negatives * 10  # Avoid infinite loop
        attempts = 0
        
        while len(negatives) < num_negatives and attempts < max_attempts:
            candidate = self.rng.randint(0, self.num_items)
            if candidate not in exclude and candidate not in negatives:
                negatives.append(candidate)
            attempts += 1
        
        if len(negatives) < num_negatives:
            # Fill remaining with any available items
            available = set(range(self.num_items)) - exclude - set(negatives)
            if available:
                remaining = self.rng.choice(
                    list(available),
                    size=min(num_negatives - len(negatives), len(available)),
                    replace=False
                )
                negatives.extend(remaining.tolist())
        
        return negatives
    
    def sample_batch(
        self,
        user_indices: torch.Tensor,
        num_negatives: int = 1,
    ) -> torch.Tensor:
        """
        Sample negative items for a batch of users.
        
        Args:
            user_indices: Tensor of user indices [batch_size]
            num_negatives: Number of negative samples per user
            
        Returns:
            Tensor of negative item indices [batch_size, num_negatives]
        """
        batch_size = user_indices.size(0)
        negatives = torch.zeros(batch_size, num_negatives, dtype=torch.long)
        
        for i, user_idx in enumerate(user_indices.tolist()):
            neg_items = self.sample(user_idx, num_negatives)
            negatives[i] = torch.tensor(neg_items[:num_negatives], dtype=torch.long)
        
        return negatives
