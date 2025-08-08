# ======================================================================================
# cql_trainer.py (Corrected for Structured State)
#
# Description:
#   This version is updated to work with the Transformer-based networks that
#   expect a structured state_dict as input, rather than a flat tensor.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
import torch.nn.functional as F
from copy import deepcopy

# Import our new networks
from src.stage_3_1_structured_mlp_cql.networks import Actor, Critic
from src.shared.utils import TrainingConfig

class CQLTrainer:
    """
    The main class for the CQL algorithm, adapted for structured state dictionaries.
    """
    def __init__(
        self,
        config: TrainingConfig, # Pass the whole config for easy access to params
        device: str,
    ):
        self.device = device
        self.config = config
        
        # --- Hyperparameters from config ---
        self.lr = config.learning_rate
        self.gamma = config.gamma
        self.tau = config.tau
        self.cql_alpha = config.cql_alpha
        self.cql_n_actions = config.cql_n_actions
        self.tune_alpha = config.tune_alpha
        self.target_entropy = -float(config.action_dim) if config.target_entropy is None else config.target_entropy

        # --- Networks ---
        self.actor = Actor(config).to(device)
        self.critic = Critic(config).to(device)
        self.critic_target = deepcopy(self.critic)

        # --- Optimizers (remain the same) ---
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
        
        # --- SAC Alpha (Temperature) ---
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        if self.config.tune_alpha:
            self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=self.config.learning_rate)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def train_step(self, batch: tuple) -> dict:
        state_dict, action, reward, next_state_dict, done = batch
        
        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1)
        
        # --- 1. Critic Update ---
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_state_dict)
            q1_target, q2_target = self.critic_target(next_state_dict, next_actions)
            q_target = torch.min(q1_target, q2_target)
            value_target = reward + self.gamma * (1.0 - done) * (q_target - self.alpha * next_log_probs)

        q1_current, q2_current = self.critic(state_dict, action)
        critic_loss_bellman = F.mse_loss(q1_current, value_target) + F.mse_loss(q2_current, value_target)
        
        # --- CQL Regularizer Calculation ---
        # NOTE: This part is tricky with structured data. We need a way to repeat the dict.
        batch_size = state_dict['ego'].shape[0]
        
        # Helper function to repeat each tensor in the state dictionary
        def repeat_state_dict(d, n):
            return {k: v.repeat_interleave(n, dim=0) for k, v in d.items()}
        
        repeated_states_dict = repeat_state_dict(state_dict, self.cql_n_actions)
        
        random_actions = self.actor.action_scale * (torch.rand(batch_size * self.cql_n_actions, self.config.action_dim, device=self.device) * 2 - 1) + self.actor.action_bias
        
        q1_random, q2_random = self.critic(repeated_states_dict, random_actions)
        q1_random = q1_random.view(batch_size, self.cql_n_actions, 1)
        q2_random = q2_random.view(batch_size, self.cql_n_actions, 1)

        policy_actions, _ = self.actor.sample(repeated_states_dict)
        q1_policy, q2_policy = self.critic(repeated_states_dict, policy_actions)
        q1_policy = q1_policy.view(batch_size, self.cql_n_actions, 1)
        q2_policy = q2_policy.view(batch_size, self.cql_n_actions, 1)

        cql_cat_q1 = torch.cat([q1_random, q1_policy], dim=1)
        cql_cat_q2 = torch.cat([q2_random, q2_policy], dim=1)
        
        log_sum_exp_q1 = torch.logsumexp(cql_cat_q1, dim=1)
        log_sum_exp_q2 = torch.logsumexp(cql_cat_q2, dim=1)
        
        cql_loss_q1 = (log_sum_exp_q1 - q1_current).mean()
        cql_loss_q2 = (log_sum_exp_q2 - q2_current).mean()
        
        critic_loss = critic_loss_bellman + self.cql_alpha * (cql_loss_q1 + cql_loss_q2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()
        
        # --- 2. Actor and Alpha Update ---
        new_actions, log_probs = self.actor.sample(state_dict)
        q1_new, q2_new = self.critic(state_dict, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()
        
        if self.tune_alpha:
            alpha_loss = (-self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
            
        # --- 3. Target Network Update ---
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)
                
        return {
            'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(), 'alpha': self.alpha,
            'cql_loss_q1': cql_loss_q1.item(),
        }

    def save(self, path: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target = deepcopy(self.critic)