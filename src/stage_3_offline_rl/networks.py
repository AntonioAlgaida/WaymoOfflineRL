# ======================================================================================
# networks.py (Stage 3 - Definitive Transformer Actor-Critic)
#
# Description:
#   This version correctly handles the final, rich, structured state dictionary,
#   including the crucial 'route' information as a sequential input to the Transformer.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.shared.utils import TrainingConfig

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class TransformerEncoder(nn.Module):
    """
    The shared Transformer backbone. Now includes an encoder for the rules vector,
    treating it as a special context token in the attention mechanism.
    """
    def __init__(self, config: TrainingConfig, embedding_dim: int = 128):
        super().__init__()
        
        # --- Input Encoders for each entity type ---
        self.ego_encoder = nn.Linear(3, embedding_dim)
        self.agent_encoder = nn.Linear(10, embedding_dim)
        self.lane_encoder = nn.Linear(2, embedding_dim)
        self.crosswalk_encoder = nn.Linear(2, embedding_dim)
        self.route_encoder = nn.Linear(2, embedding_dim)
        self.rule_encoder = nn.Linear(9, embedding_dim) # NEW: Encoder for the rules vector
        
        # --- Transformer Layers ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode all entities, including the rules
        ego_embedding = self.ego_encoder(state_dict['ego'])
        agent_embeddings = self.agent_encoder(state_dict['agents'])
        lane_embeddings = self.lane_encoder(state_dict['lanes'])
        crosswalk_embeddings = self.crosswalk_encoder(state_dict['crosswalks'])
        route_embeddings = self.route_encoder(state_dict['route'])
        rule_embedding = self.rule_encoder(state_dict['rules']) # NEW: Encode the rules
        
        # Create the full sequence for the Transformer.
        # We now have two special tokens at the beginning: ego and rules.
        full_sequence = torch.cat([
            ego_embedding.unsqueeze(1),
            rule_embedding.unsqueeze(1), # NEW: Add rule embedding to the sequence
            agent_embeddings,
            lane_embeddings,
            crosswalk_embeddings,
            route_embeddings
        ], dim=1)
        # Sequence length = 1 + 1 + 15 + 50 + 10 + 10 = 87
        
        transformer_output = self.transformer(full_sequence)
        
        # The output of the "ego" token is now aware of the rules.
        scene_embedding = transformer_output[:, 0, :]
        return scene_embedding

class TransformerCritic(nn.Module):
    def __init__(self, config: TrainingConfig, action_dim: int = 2):
        super().__init__()
        embedding_dim = 64
        self.scene_encoder = TransformerEncoder(config, embedding_dim)
        
        # The input to the Q-head is now simpler: just the scene embedding and the action.
        # The rules are already incorporated into the scene_embedding.
        q_head_input_dim = embedding_dim + action_dim
        
        self.q1_head = nn.Sequential(
            nn.Linear(q_head_input_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 1)
        )
        self.q2_head = nn.Sequential(
            nn.Linear(q_head_input_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

    def forward(self, state_dict: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scene_embedding = self.scene_encoder(state_dict)
        # Concatenate the final scene summary with the action
        x = torch.cat([scene_embedding, action], dim=1)
        q1 = self.q1_head(x)
        q2 = self.q2_head(x)
        return q1, q2

    def Q1(self, state_dict: dict, action: torch.Tensor) -> torch.Tensor:
        scene_embedding = self.scene_encoder(state_dict)
        x = torch.cat([scene_embedding, action], dim=1)
        q1 = self.q1_head(x)
        return q1

class TransformerActor(nn.Module):
    def __init__(self, config: TrainingConfig, action_dim: int = 2):
        super().__init__()
        embedding_dim = 64
        
        self.scene_encoder = TransformerEncoder(config, embedding_dim)
        
        # Correctly size the input for the Actor-head
        actor_head_input_dim = embedding_dim
        
        self.actor_head = nn.Sequential(
            nn.Linear(actor_head_input_dim, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.LayerNorm(64)
        )
        self.mean_head = nn.Linear(64, action_dim)
        self.log_std_head = nn.Linear(64, action_dim)
        
        # Action Rescaling
        action_scale = torch.tensor([(config.MAX_ACCELERATION - config.MIN_ACCELERATION) / 2.0, (config.MAX_STEERING_ANGLE * 2) / 2.0])
        action_bias = torch.tensor([(config.MAX_ACCELERATION + config.MIN_ACCELERATION) / 2.0, 0.0])
        self.register_buffer('action_scale', action_scale)
        self.register_buffer('action_bias', action_bias)

    def forward(self, state_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # The scene_embedding now contains all necessary context, including rules.
        scene_embedding = self.scene_encoder(state_dict)
        x = self.actor_head(scene_embedding)
        
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, state_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state_dict)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob