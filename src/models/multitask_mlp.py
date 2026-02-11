import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskClinicalMLP(nn.Module):
    """
    Multi-Task MLP Decoder for Clinical Trial Phase Prediction.
    
    Predicts the probability of achieving each of 4 max clinical trial phases/outcomes:
    - pos
    - unmet
    - adv
    - op
    
    Inputs: Concatenated node embeddings [h_u || h_v]
    Outputs: 4 independent probability scores in [0, 1]
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task head: 4 outputs
        # GATher uses ReLU activation for regression tasks (continuous phase values)
        self.head = nn.Linear(hidden_dim // 2, 4)
        self.output_activation = nn.ReLU()  # Non-negative continuous predictions
        
    def forward(self, disease_emb, target_emb):
        """
        Args:
            disease_emb: [batch_size, dim]
            target_emb: [batch_size, dim]
            
        Returns:
            Dictionary of logits for each task {'pos', 'unmet', 'adv', 'op'}
        """
        # Concatenate embeddings
        x = torch.cat([disease_emb, target_emb], dim=-1)
        
        # Shared encoder
        feat = self.net(x)
        
        # Multi-task prediction (continuous phase values)
        # Apply ReLU for regression (GATher approach)
        outputs = self.head(feat)
        outputs = self.output_activation(outputs)  # Non-negative predictions
        
        return {
            'pos': outputs[:, 0],
            'unmet': outputs[:, 1],
            'adv': outputs[:, 2],
            'op': outputs[:, 3]
        }

class WeightedMultiTaskLoss(nn.Module):
    """
    Weighted Multi-Task Loss for Regression (MSE or Huber).
    
    GATher uses regression losses for continuous phase prediction.
    """
    def __init__(self, weights=None, use_huber=True):
        super().__init__()
        self.weights = weights if weights else {'pos': 1.0, 'unmet': 1.0, 'adv': 1.0, 'op': 1.0}
        self.use_huber = use_huber  # Default to Huber (more robust)
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict of logits
            targets: Dict of float targets
            
        Returns:
            total_loss, dict_of_task_losses
        """
        total_loss = 0
        task_losses = {}
        
        for task in ['pos', 'unmet', 'adv', 'op']:
            pred = predictions[task]
            target = targets[task]
            
            # Regression losses (GATher approach)
            # Predictions are already ReLU-activated (non-negative)
            # Targets are continuous phase values (0.0 to 4.0 or normalized)
            if self.use_huber:
                # Huber loss (robust to outliers)
                loss = F.huber_loss(pred, target, delta=1.0)
            else:
                # Mean Squared Error
                loss = F.mse_loss(pred, target)
                
            w_loss = loss * self.weights.get(task, 1.0)
            total_loss += w_loss
            task_losses[task] = loss.item() # Log raw loss
            
        return total_loss, task_losses

def compute_task_weights(labels_df, inverse_freq=True):
    """
    Compute task weights based on label frequencies.
    """
    tasks = ['y_pos', 'y_unmet', 'y_adv', 'y_op']
    weights = {}
    
    total = len(labels_df)
    
    for task in tasks:
        task_key = task.replace('y_', '')
        
        if inverse_freq:
            pos_count = labels_df[task].sum()
            # Simple inverse frequency
            if pos_count > 0:
                w = total / (2 * pos_count) # Heuristic for balancing
            else:
                w = 1.0
            weights[task_key] = w
        else:
            weights[task_key] = 1.0
            
    return weights
