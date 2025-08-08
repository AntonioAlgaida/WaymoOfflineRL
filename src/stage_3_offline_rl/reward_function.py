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
W_PROGRESS = 1.0
W_SAFETY = -2.0
W_COMFORT = -2.0
W_LANE = -1.0

# --- Safety/Reward Constants ---
MIN_TTC_THRESHOLD = 2.5
LATERAL_INFLUENCE_THRESHOLD = 2.0
TARGET_SPEED = 15.0 # m/s
MAX_LANE_DEVIATION_PENALTY = 5.0

def _compute_progress_reward(ego_features: np.ndarray, route_features: np.ndarray) -> float:
    """
    Calculates a reward for making progress along the intended route.
    This is more robust than rewarding raw speed, as it specifically rewards
    movement towards the next waypoint on the planned path.

    Args:
        ego_features: Ego vector [speed, accel, yaw_rate].
        route_features: A (10, 2) NumPy array of ego-centric future waypoints.

    Returns:
        A positive scalar reward for on-route progress.
    """
    current_speed = ego_features[0]
    
    # The SDC's current velocity vector in its own reference frame is always
    # pointing forward along the x-axis.
    ego_velocity_vector = np.array([current_speed, 0.0])

    # The direction to the immediate next waypoint on the route.
    # We can use the first waypoint (index 0) as our immediate target.
    # A more advanced version could use a point further along the path.
    target_route_direction = route_features[0] # Shape (2,)
    
    # Normalize the direction vector to get a unit vector
    norm = np.linalg.norm(target_route_direction)
    # If the waypoint is at our current location (norm is zero), there's no progress to be made.
    if norm < 1e-4:
        return 0.0
    
    unit_route_direction = target_route_direction / norm
    
    # Project the ego's velocity vector onto the route direction vector.
    # This gives us the component of our speed that is aligned with the desired path.
    # The result is a scalar value.
    projected_speed = np.dot(ego_velocity_vector, unit_route_direction)
    
    # We only reward positive projected speed (moving towards the waypoint).
    # If the agent is moving away from the waypoint, this value will be negative,
    # and max(0, ...) will correctly result in a zero progress reward.
    on_route_progress = max(0, projected_speed)
    
    # Normalize by the target speed and cap, as before.
    reward = min(on_route_progress / TARGET_SPEED, 1.2)
    
    return reward

def _compute_safety_penalty(agent_features: np.ndarray) -> float:
    """
    Calculates a speed-dependent safety penalty based on Time-to-Collision (TTC).
    This function primarily penalizes the risk of forward collisions (tailgating).

    Args:
        agent_features: A (15, 10) NumPy array for the closest agents.

    Returns:
        A positive penalty value (0 for safe, >0 for unsafe), which will be
        made negative by its weight W_SAFETY.
    """
    # Initialize a list to store TTC values for relevant agents
    ttc_values = []

    # agent_features columns: [rel_x, rel_y, rel_vx, rel_vy, rel_head, len, wid, t1, t2, t3]
    for agent in agent_features:
        # Check if the agent is valid (padding will have zero position)
        if np.linalg.norm(agent[:2]) < 1e-4:
            continue

        rel_x, rel_y = agent[0], agent[1]
        rel_vx, rel_vy = agent[2], agent[3]

        # --- Filtering Logic ---
        # 1. Is the agent in front of us? (rel_x > 0)
        # 2. Is the agent laterally close? (abs(rel_y) is small)
        # 3. Are we actually getting closer to it? (rel_vx < 0)
        if rel_x > 0 and abs(rel_y) < LATERAL_INFLUENCE_THRESHOLD and rel_vx < -0.1:
            
            # This is the longitudinal relative speed (how fast we are closing the gap)
            closing_speed = -rel_vx
            
            # Longitudinal distance, accounting for our own length
            # We use the front of our car to the back of their car
            # For simplicity, we'll just use rel_x for now. A more advanced
            # version would use bounding box edges.
            longitudinal_distance = rel_x

            # Calculate TTC
            ttc = longitudinal_distance / closing_speed
            ttc_values.append(ttc)

    # If no agents are a threat, there is no penalty
    if not ttc_values:
        return 0.0

    # The danger is determined by the *minimum* TTC to any threatening agent
    min_ttc = np.min(ttc_values)

    # If the minimum TTC is above our safety threshold, the situation is safe
    if min_ttc > MIN_TTC_THRESHOLD:
        return 0.0
    
    # --- Penalty Calculation ---
    # The penalty increases quadratically as TTC drops from the threshold to zero.
    # When min_ttc == MIN_TTC_THRESHOLD, penalty is 0.
    # When min_ttc == 0, penalty is 1.
    penalty = (1.0 - (min_ttc / MIN_TTC_THRESHOLD))**2
    
    return penalty

def _compute_comfort_penalty(ego_features: np.ndarray) -> float:
    """
    Calculates a penalty for uncomfortable actions (high acceleration/jerk, sharp turns).
    Args:
        ego_features: A (3,) NumPy array of [speed, acceleration, yaw_rate].
    Returns:
        A negative scalar penalty.
    """
    acceleration = ego_features[1]
    yaw_rate = ego_features[2]
    
    # Penalize extreme acceleration/deceleration and sharp turns.
    # The normalization constants act as a "soft cap".
    # For example, a yaw_rate of 0.8 rad/s (a sharp turn) results in a
    # penalty component of (0.8/0.8)^2 = 1.0. A more extreme yaw_rate of 1.6
    # would result in a much larger penalty of (1.6/0.8)^2 = 4.0.
    accel_penalty = (acceleration / 5.0)**2
    yaw_penalty = (yaw_rate / 0.8)**2
    
    return accel_penalty + yaw_penalty

    
    # # Penalize extreme acceleration/deceleration and sharp turns.
    # # We square them to penalize larger values more heavily.
    # accel_penalty = (acceleration / 5.0)**2  # Normalize by a reasonable max accel
    # yaw_penalty = (yaw_rate / 0.8)**2   # Normalize by a reasonable max yaw_rate
    
    # return min(accel_penalty + yaw_penalty, 1.0)

def _compute_lane_adherence_penalty(map_features: np.ndarray) -> float:
    """
    Calculates a penalty for deviating from the center of the nearest lane.
    Args:
        map_features: A (120,) flattened NumPy array of map features.
                      The first 100 elements are the 50 closest lane points.
    Returns:
        A negative scalar penalty.
    """
    # The lane points are already sorted by distance in the feature vector.
    # The closest point is the first one. Its features are relative_x, relative_y.
    closest_lane_point_rel_y = map_features[0][1] # The y-component is the lateral deviation
    
    # The penalty is the squared lateral distance from the lane center.
    lane_deviation_penalty = closest_lane_point_rel_y**2
        
    # Scale the penalty. Division by 25 means a 5m deviation (5^2=25) gives a penalty of 1.
    scaled_penalty = lane_deviation_penalty / 25.0
    
    return min(scaled_penalty, MAX_LANE_DEVIATION_PENALTY)

# --- The Main, Config-Driven Reward Function ---
def compute_reward(state_dict: np.ndarray) -> float:
    """
    The main reward function. It computes a scalar reward for a given state.
    
    This function is now driven by a config object to robustly deconstruct
    the state vector.
    
    Args:
        state_dict: A structured dictionary containing the state of the environment.
        
    Returns:
        A scalar float representing the total reward for being in that state.
    """
    # --- 1. Deconstruct the State Dictionary ---
    # This is now a simple, readable dictionary lookup. No more slicing!
    ego_features = state_dict['ego']
    agent_features = state_dict['agents']
    lane_features = state_dict['lanes']
    route_features = state_dict['route'] # Get the new route features

    # rules_features = state_dict['rules'] # Available if needed

    # --- 2. Calculate Each Reward Component ---
    progress_reward = _compute_progress_reward(ego_features, route_features)
    safety_penalty = _compute_safety_penalty(agent_features)
    comfort_penalty = _compute_comfort_penalty(ego_features)
    lane_adherence_penalty = _compute_lane_adherence_penalty(lane_features)
    
    # --- 3. Combine with Weights ---
    total_reward = (
        W_PROGRESS * progress_reward +
        W_SAFETY * safety_penalty +
        W_COMFORT * comfort_penalty +
        W_LANE * lane_adherence_penalty
    )
        
    # Print debug if abs(total_reward) > 10:
    # if total_reward > 10:
    #     print(f"DEBUG: Total Reward: {total_reward:.4f}")
    #     print(f"  Progress: {progress_reward:.4f}, Safety Penalty: {safety_penalty:.4f}, "
    #           f"Comfort Penalty: {comfort_penalty:.4f}, Lane Adherence Penalty: {lane_adherence_penalty:.4f}")
    
    # if total_reward < -10:
    #     print(f"DEBUG: Total Reward: {total_reward:.4f}")
    #     print(f"  Progress: {progress_reward:.4f}, Safety Penalty: {safety_penalty:.4f}, "
    #           f"Comfort Penalty: {comfort_penalty:.4f}, Lane Adherence Penalty: {lane_adherence_penalty:.4f}")
    
    return total_reward
