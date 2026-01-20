#!/usr/bin/env python3
"""
PyTorch Lightning wrapper for HGT link prediction.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from .hgt import HGTLinkPredictor


class HGTRecLightning(pl.LightningModule):
    """
    PyTorch Lightning module for HGT-based link prediction.
    """
    
    def __init__(
        self,
        model: HGTLinkPredictor,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        supervision_src_type: str = "disease",
        supervision_dst_type: str = "target",
    ):
        """
        Initialize Lightning module.
        
        Args:
            model: HGTLinkPredictor model
            lr: Learning rate
            weight_decay: Weight decay
            supervision_src_type: Source node type for supervision
            supervision_dst_type: Destination node type for supervision
        """
        super().__init__()
        
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.supervision_src_type = supervision_src_type
        self.supervision_dst_type = supervision_dst_type
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        edge_label_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        return self.model(
            x_dict,
            edge_index_dict,
            edge_label_index,
            self.supervision_src_type,
            self.supervision_dst_type,
        )
    
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Unpack batch
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_label_index = batch['disease', 'clinical_trial', 'target'].edge_label_index
        edge_label = batch['disease', 'clinical_trial', 'target'].edge_label
        
        # Forward pass
        pred = self(x_dict, edge_index_dict, edge_label_index)
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)
        
        # Log
        self.log('train_loss', loss, prog_bar=True, batch_size=edge_label.size(0))
        
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Unpack batch
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_label_index = batch['disease', 'clinical_trial', 'target'].edge_label_index
        edge_label = batch['disease', 'clinical_trial', 'target'].edge_label
        
        # Forward pass
        pred = self(x_dict, edge_index_dict, edge_label_index)
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)
        
        # Log
        self.log('val_loss', loss, prog_bar=True, batch_size=edge_label.size(0))
        
        return loss
    
    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        # Unpack batch
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_label_index = batch['disease', 'clinical_trial', 'target'].edge_label_index
        edge_label = batch['disease', 'clinical_trial', 'target'].edge_label
        
        # Forward pass
        pred = self(x_dict, edge_index_dict, edge_label_index)
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(pred, edge_label)
        
        # Log
        self.log('test_loss', loss, batch_size=edge_label.size(0))
        
        return {
            'predictions': pred,
            'labels': edge_label,
            'edge_index': edge_label_index,
        }
    
    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
