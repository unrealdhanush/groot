# src/modeling/fusion_model.py

"""
Defines the Fusion Model:
1. Combines structured features and text embeddings.
2. Uses fully connected layers for prediction.
"""

import torch
import torch.nn as nn

class FusionModel(nn.Module):
    """
    Fusion model combining structured features and text embeddings for classification.
    """
    def __init__(self, structured_input_dim, embedding_dim, hidden_dims, dropout_rate=0.2):
        """
        Args:
            structured_input_dim (int): Number of structured input features.
            embedding_dim (int): Size of text embedding vectors.
            hidden_dims (list): List of hidden layer sizes for the feed-forward network.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(FusionModel, self).__init__()
        
        # Separate branches for structured and text embeddings
        self.structured_branch = nn.Sequential(
            nn.Linear(structured_input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.text_branch = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined branch
        self.combined_branch = nn.Sequential(
            nn.Linear(2 * hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, structured_inputs, text_embeddings):
        """
        Forward pass for the model.
        
        Args:
            structured_inputs (torch.Tensor): Structured input features [batch_size, structured_input_dim].
            text_embeddings (torch.Tensor): Text embeddings [batch_size, embedding_dim].
        
        Returns:
            torch.Tensor: Predicted probabilities for binary classification.
        """
        structured_out = self.structured_branch(structured_inputs)
        text_out = self.text_branch(text_embeddings)
        
        combined = torch.cat((structured_out, text_out), dim=1)
        output = self.combined_branch(combined)
        
        return output