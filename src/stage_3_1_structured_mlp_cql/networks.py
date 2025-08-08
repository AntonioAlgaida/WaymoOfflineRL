# ======================================================================================
# networks.py (Stage 3.1 - Structured MLP Actor-Critic)
# ======================================================================================
import torch
import torch.nn as nn
from torch.distributions import Normal

from src.shared.utils import TrainingConfig

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class StructuredMLP(nn.Module):
    """A shared MLP backbone that first flattens the structured state dict."""
    def __init__(self, config: TrainingConfig):
        super().__init__()
        # Calculate the total flattened input dimension
        ego_dim, agent_dim = 3, config.num_closest_agents * 10
        lane_dim, cw_dim = config.num_closest_map_points * 2, config.num_closest_crosswalk_points * 2
        route_dim, rules_dim = config.num_future_waypoints * 2, 9
        self.total_input_dim = ego_dim + agent_dim + lane_dim + cw_dim + route_dim + rules_dim
        
        # --- The MLP Architecture ---
        self.mlp_body = nn.Sequential(
            nn.Linear(self.total_input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.LayerNorm(256),
        )
        

    def forward(self, state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        flat_state = torch.cat([
            state_dict['ego'], state_dict['agents'].flatten(1),
            state_dict['lanes'].flatten(1), state_dict['crosswalks'].flatten(1),
            state_dict['route'].flatten(1), state_dict['rules']
        ], dim=1)
        
        # Verify the flattened shape
        # assert flat_state.shape[1] == self.total_input_dim
        
        return self.mlp_body(flat_state)

class Critic(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.backbone = StructuredMLP(config)
        
        # Q-heads take the backbone's output concatenated with the action
        q_head_input_dim = 256 + config.action_dim
        
        self.q1_head = nn.Sequential(nn.Linear(q_head_input_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2_head = nn.Sequential(nn.Linear(q_head_input_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state_dict: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state_embedding = self.backbone(state_dict)
        x = torch.cat([state_embedding, action], dim=1)
        return self.q1_head(x), self.q2_head(x)

    def Q1(self, state_dict: dict, action: torch.Tensor) -> torch.Tensor:
        state_embedding = self.backbone(state_dict)
        x = torch.cat([state_embedding, action], dim=1)
        return self.q1_head(x)

class Actor(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.backbone = StructuredMLP(config)
        
        # Action heads take the backbone's output
        actor_head_input_dim = 256
        self.mean_head = nn.Linear(actor_head_input_dim, config.action_dim)
        self.log_std_head = nn.Linear(actor_head_input_dim, config.action_dim)
        
        # Action Rescaling
        action_scale = torch.tensor([(config.MAX_ACCELERATION - config.MIN_ACCELERATION) / 2.0, (config.MAX_STEERING_ANGLE * 2) / 2.0])
        action_bias = torch.tensor([(config.MAX_ACCELERATION + config.MIN_ACCELERATION) / 2.0, 0.0])
        self.register_buffer('action_scale', action_scale)
        self.register_buffer('action_bias', action_bias)

    def forward(self, state_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        state_embedding = self.backbone(state_dict)
        mean = self.mean_head(state_embedding)
        log_std = self.log_std_head(state_embedding)
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