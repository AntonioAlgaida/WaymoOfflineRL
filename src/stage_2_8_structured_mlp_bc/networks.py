# In src/stage_2_8_structured_mlp_bc/networks.py
import torch
import torch.nn as nn
from src.shared.utils import TrainingConfig

class StructuredMLP_BC_Model(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        # --- Calculate the total input dimension from the structured state ---
        # This makes the model robust to changes in the config
        ego_dim = 3
        agent_dim = config.num_closest_agents * 10
        lane_dim = config.num_closest_map_points * 2
        crosswalk_dim = config.num_closest_crosswalk_points * 2
        route_dim = config.num_future_waypoints * 2
        rules_dim = 9 # As per our final spec
        
        total_input_dim = ego_dim + agent_dim + lane_dim + crosswalk_dim + route_dim + rules_dim
        
        # --- The MLP Architecture ---
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
        )
        
        # --- The Action Head with Tanh and Rescaling (Crucial!) ---
        self.action_head = nn.Sequential(
            nn.Linear(128, 2),
            nn.Tanh()
        )
        
        # Action Rescaling
        action_scale = torch.tensor([(config.MAX_ACCELERATION - config.MIN_ACCELERATION) / 2.0, (config.MAX_STEERING_ANGLE * 2) / 2.0])
        action_bias = torch.tensor([(config.MAX_ACCELERATION + config.MIN_ACCELERATION) / 2.0, 0.0])
        self.register_buffer('action_scale', action_scale)
        self.register_buffer('action_bias', action_bias)

    def forward(self, state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        # --- 1. Flatten the structured state into a single vector ---
        # The flatten(1) is important to preserve the batch dimension
        flat_state = torch.cat([
            state_dict['ego'],
            state_dict['agents'].flatten(1),
            state_dict['lanes'].flatten(1),
            state_dict['crosswalks'].flatten(1),
            state_dict['route'].flatten(1),
            state_dict['rules']
        ], dim=1)
        
        # --- 2. Pass through the MLP and Action Head ---
        x = self.mlp(flat_state)
        norm_action = self.action_head(x)
        final_action = norm_action * self.action_scale + self.action_bias
        
        return final_action