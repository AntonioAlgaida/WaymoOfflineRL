# ======================================================================================
# networks.py (Stage 2.7 - Transformer BC)
#
# Description:
#   Defines the Transformer-based model architecture for the BC-T agent.
#   This model uses self-attention to explicitly reason about the relationships
#   between the ego vehicle, other agents, and map features, creating a rich,
#   contextualized representation of the scene for decision-making.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
import torch.nn as nn
import numpy as np

# Import the config to get feature dimensions and action ranges
from src.shared.utils import TrainingConfig

class TransformerBCModel(nn.Module):
    """
    A Behavioral Cloning model that uses a Transformer Encoder to process
    a structured, entity-centric state representation.
    """
    def __init__(self, config: TrainingConfig):
        super().__init__()
        
        # --- Model Hyperparameters ---
        embedding_dim = 128      # The internal dimension of the model
        nhead = 4                # Number of self-attention heads
        num_encoder_layers = 3   # Number of stacked Transformer encoder layers
        dim_feedforward = 512    # The dimension of the feed-forward networks in the Transformer
        dropout = 0.1            # Dropout rate

        # --- 1. Input Encoders ---
        # These are simple linear layers that project the raw feature vectors for each
        # entity type into the model's high-dimensional embedding space.
        self.ego_encoder = nn.Linear(3, embedding_dim)
        self.agent_encoder = nn.Linear(10, embedding_dim)
        self.lane_encoder = nn.Linear(2, embedding_dim)
        self.crosswalk_encoder = nn.Linear(2, embedding_dim)
        # Note: Rule features are not encoded this way; they are concatenated at the end.
        
        # --- 2. The Transformer Encoder ---
        # We use a standard PyTorch TransformerEncoder.
        # It's composed of multiple TransformerEncoderLayer instances.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # This is crucial for our data shape: (Batch, Sequence, Features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # --- 3. Final MLP Head ---
        # This head takes the final scene representation from the Transformer
        # and makes the final action prediction. We also include the rule features here.
        # The input is the output of the transformer (embedding_dim) plus the rule features.
        total_input_dim = embedding_dim + 6 # 6 rule features
        
        self.head = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # --- 4. Action Head with Tanh Activation ---
        # This final layer squashes the output to the normalized range [-1, 1].
        self.action_head = nn.Sequential(
            nn.Linear(256, 2), # Outputs 2 values for [acceleration, steering]
            nn.Tanh()
        )
        
        # --- 5. Action Rescaling ---
        # We define the scale and center (bias) to map the [-1, 1] output
        # to the correct physical action range defined in our filtering step.
        MAX_ACCELERATION = 8.0  # m/s^2, very strong acceleration
        MIN_ACCELERATION = -10.0 # m/s^2, emergency braking
        MAX_STEERING_ANGLE = 0.8 # radians, ~45 degrees, a very sharp turn
        action_scale = torch.tensor([
            (MAX_ACCELERATION - MIN_ACCELERATION) / 2.0,
            (MAX_STEERING_ANGLE - (-MAX_STEERING_ANGLE)) / 2.0
        ])
        action_bias = torch.tensor([
            (MAX_ACCELERATION + MIN_ACCELERATION) / 2.0,
            (MAX_STEERING_ANGLE + (-MAX_STEERING_ANGLE)) / 2.0
        ])
        
        # Register as buffers so they are automatically moved to the correct device (e.g., GPU).
        self.register_buffer('action_scale', action_scale)
        self.register_buffer('action_bias', action_bias)

    def forward(self, state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs the forward pass of the model.
        
        Args:
            state_dict: A dictionary of tensors, e.g.,
                        {'ego': (B, 3), 'agents': (B, 15, 10), ...}
        
        Returns:
            A tensor of predicted actions of shape (B, 2).
        """
        # --- 1. Encode all entities into the embedding space ---
        ego_embedding = self.ego_encoder(state_dict['ego'])             # Shape: (B, D)
        agent_embeddings = self.agent_encoder(state_dict['agents'])     # Shape: (B, 15, D)
        lane_embeddings = self.lane_encoder(state_dict['lanes'])       # Shape: (B, 50, D)
        crosswalk_embeddings = self.crosswalk_encoder(state_dict['crosswalks']) # Shape: (B, 10, D)
        
        # --- 2. Prepare the sequence for the Transformer ---
        # We treat the ego vehicle's embedding as a special token (like [CLS] in BERT).
        # We concatenate all entity embeddings into one long sequence.
        # The Transformer will learn the relationships between all these items.
        full_sequence = torch.cat([
            ego_embedding.unsqueeze(1),  # Add a sequence dimension -> (B, 1, D)
            agent_embeddings,
            lane_embeddings,
            crosswalk_embeddings
        ], dim=1) # Final sequence shape: (B, 1 + 15 + 50 + 10, D) = (B, 76, D)
        
        # --- 3. Pass through the Transformer Encoder ---
        # The self-attention mechanism processes the entire sequence.
        transformer_output = self.transformer_encoder(full_sequence)
        
        # --- 4. Aggregate the Scene Information ---
        # We take the output embedding corresponding to our initial ego token.
        # After passing through the Transformer, this token has gathered contextual
        # information from all other entities in the scene via self-attention.
        # It now represents a rich summary of the entire scene.
        scene_embedding = transformer_output[:, 0, :] # Shape: (B, D)
        
        # --- 5. Final Prediction ---
        # Concatenate the scene summary with the non-spatial rule features.
        final_embedding = torch.cat([scene_embedding, state_dict['rules']], dim=1)
        
        # Pass this final representation through the MLP head.
        x = self.head(final_embedding)
        
        # Get the normalized action from the action head.
        norm_action = self.action_head(x)
        
        # Rescale the action to the correct physical range.
        final_action = norm_action * self.action_scale + self.action_bias
        
        return final_action