# ======================================================================================
# reward_function.py (Version 4.0 - Fully Structured)
#
# Description:
#   Defines the multi-objective reward function for the Offline RL agent.
#   This version operates directly on the structured state dictionary, making
#   the code cleaner, more robust, and easier to debug.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import numpy as np
# The TrainingConfig is no longer needed here, but the RLDataset will still use it.

# --- Reward Component Weights ---
# These weights now define the *relative importance* of each component
# before the final normalization step.
W_ROUTE_FOLLOWING = 2.0  # Give more weight to the primary objective
W_SAFETY = -5.0
W_COMFORT = -3.0

# --- Reward Scaling Constant ---
# This controls the sensitivity of the final tanh normalization.
# A smaller value makes the function saturate faster.
REWARD_SCALING_FACTOR = 10.0

# --- Safety/Reward Constants ---
MIN_TTC_THRESHOLD = 2.5
LATERAL_INFLUENCE_THRESHOLD = 2.0

def _compute_route_following_reward(ego_features: np.ndarray, route_features: np.ndarray) -> float:
    """
    Calculates a unified reward for progress and adherence, now returning a raw score.
    """
    current_speed = ego_features[0]
    
    ego_velocity_vector = np.array([current_speed, 0.0])
    target_route_direction = route_features[0]
    norm = np.linalg.norm(target_route_direction)
    if norm < 1e-4: return 0.0
    
    unit_route_direction = target_route_direction / norm
    projected_speed = np.dot(ego_velocity_vector, unit_route_direction)
    
    # Progress score is now the raw on-route speed
    progress_score = max(0, projected_speed)
    
    # Adherence (cross-track error) penalty
    cross_track_error = abs(target_route_direction[1])
    # A 5m deviation results in a penalty of -25
    adherence_penalty_score = -(cross_track_error**2)
    
    # Return a combined raw score
    return progress_score + adherence_penalty_score

def _compute_safety_penalty(agent_features: np.ndarray) -> float:
    # This function already returns a value in [0, 1], which is good.
    # We will keep it as is. The W_SAFETY will scale its importance.
    ttc_values = []
    for agent in agent_features:
        if np.linalg.norm(agent[:2]) < 1e-4: continue
        rel_x, rel_y, rel_vx = agent[0], agent[1], agent[2]
        if rel_x > 0 and abs(rel_y) < LATERAL_INFLUENCE_THRESHOLD and rel_vx < -0.1:
            closing_speed = -rel_vx
            ttc = rel_x / closing_speed
            ttc_values.append(ttc)
    if not ttc_values: return 0.0
    min_ttc = np.min(ttc_values)
    if min_ttc > MIN_TTC_THRESHOLD: return 0.0
    penalty = (1.0 - (min_ttc / MIN_TTC_THRESHOLD))**2
    return penalty

def _compute_comfort_penalty(ego_features: np.ndarray) -> float:
    # This function can produce large values, let's keep it uncapped for now.
    acceleration = ego_features[1]
    yaw_rate = ego_features[2]
    accel_penalty = (acceleration / 5.0)**2
    yaw_penalty = (yaw_rate / 0.8)**2
    return accel_penalty + yaw_penalty

# --- The Main, Final Reward Function ---
def compute_reward(state_dict: dict) -> float:
    """
    Computes a final scalar reward, scaled to the range [-1, 1].
    """
    ego_features = state_dict['ego']
    agent_features = state_dict['agents']
    route_features = state_dict['route']

    # --- 1. Calculate Raw Scores for Each Component ---
    route_score = _compute_route_following_reward(ego_features, route_features)
    safety_penalty = _compute_safety_penalty(agent_features)
    comfort_penalty = _compute_comfort_penalty(ego_features)
    
    # --- 2. Combine with Weights to get a single raw score ---
    raw_total_reward = (
        W_ROUTE_FOLLOWING * route_score +
        W_SAFETY * safety_penalty +
        W_COMFORT * comfort_penalty
    )
    
    # --- 3. Scale and Squash the final reward to [-1, 1] ---
    # This is the crucial normalization step.
    normalized_reward = np.tanh(raw_total_reward / REWARD_SCALING_FACTOR)
    
    return normalized_reward
