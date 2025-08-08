# In src/stage_2_6_structured_bc/networks.py
import torch
import torch.nn as nn

class EntityEncoder(nn.Module):
    """A simple MLP to encode a single entity's features."""
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.model(x)

class StructuredBCModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_dim = 128
        
        # --- Encoders for each entity type ---
        self.ego_encoder = EntityEncoder(input_dim=3, output_dim=embedding_dim)
        self.agent_encoder = EntityEncoder(input_dim=10, output_dim=embedding_dim)
        self.lane_encoder = EntityEncoder(input_dim=2, output_dim=embedding_dim)
        self.crosswalk_encoder = EntityEncoder(input_dim=2, output_dim=embedding_dim)
        
        total_embedding_dim = embedding_dim * 4 + 6

        # --- The Main MLP Head (without the final activation) ---
        self.head = nn.Sequential(
            nn.Linear(total_embedding_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # --- NEW: Final Action Head ---
        # A separate final layer followed by Tanh
        self.action_head = nn.Sequential(
            nn.Linear(256, 2),
            nn.Tanh() # This squashes the output to the range [-1, 1]
        )
        
        # --- NEW: Action Scaling and Biasing ---
        # We define the scale and center (bias) for our action space
        # Accel: range [-10, 8] -> center = -1, scale = 9
        # Steer: range [-0.8, 0.8] -> center = 0, scale = 0.8
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
        
        # Register as buffers so they are moved to the correct device (e.g., GPU)
        self.register_buffer('action_scale', action_scale)
        self.register_buffer('action_bias', action_bias)

    def forward(self, state_dict):
        # --- Process each entity set ---
        ego_embedding = self.ego_encoder(state_dict['ego'])
        
        # agent_encoder is applied to each of the 15 agents
        agent_embeddings = self.agent_encoder(state_dict['agents'])
        
        # lane_encoder is applied to each of the 50 lane points
        lane_embeddings = self.lane_encoder(state_dict['lanes'])
        
        # crosswalk_encoder is applied to each of the 10 crosswalk points
        crosswalk_embeddings = self.crosswalk_encoder(state_dict['crosswalks'])

        # --- Aggregate with Max Pooling (provides permutation invariance) ---
        # We take the element-wise maximum across the entity dimension
        agg_agent_embedding, _ = torch.max(agent_embeddings, dim=1)
        agg_lane_embedding, _ = torch.max(lane_embeddings, dim=1)
        agg_crosswalk_embedding, _ = torch.max(crosswalk_embeddings, dim=1)
        
        # --- Concatenate all features for the final decision ---
        combined_features = torch.cat([
            ego_embedding,
            agg_agent_embedding,
            agg_lane_embedding,
            agg_crosswalk_embedding,
            state_dict['rules'] # Pass the rule features directly
        ], dim=1)
        
        # Pass through the main body of the MLP
        x = self.head(combined_features)
        
        # Pass through the final action head to get a normalized action in [-1, 1]
        norm_action = self.action_head(x)
        
        # --- NEW: Rescale the normalized action to the correct physical range ---
        final_action = norm_action * self.action_scale + self.action_bias
        
        return final_action