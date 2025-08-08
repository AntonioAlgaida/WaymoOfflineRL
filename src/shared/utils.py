# ======================================================================================
# utils.py
#
# Description:
#   A central repository for shared functions, classes, and configurations used
#   across different scripts in the WaymoOfflineAgent project.
#
# Author: Antonio Guillen-Perez
# Date: 2025-07-31
# ======================================================================================

import os
import torch
import numpy as np
import jax.numpy as jnp
from waymax import datatypes
from dataclasses import dataclass

import torch
import torch.nn as nn

def normalize_angle(angle):
    # This helper function is the same
    return (angle + np.pi) % (2 * np.pi) - np.pi

@dataclass
class TrainingConfig:
    """
    A single dataclass to hold all configuration parameters for the project.
    """
    # ==========================================================================
    # --- 1. Path Configurations ---
    # ==========================================================================
    # Raw parsed data
    train_data_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/processed_training')
    val_data_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/processed_validation')
    
    # Stage 2 (Displacement BC) Preprocessed Data
    bc_preprocess_train_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/bc_preprocessed_training')
    bc_preprocess_val_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/bc_preprocessed_validation')
    
    # Stage 2.5 (Kinematic BC) Preprocessed Data
    bck_preprocess_train_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/bc_kinematic_preprocessed_training')
    bck_preprocess_val_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/bc_kinematic_preprocessed_validation')
    
    # Stage 2.6 (Structured BC) CHUNKED Preprocessed Data
    bcs_preprocess_train_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/bc_structured_chunked_training')
    bcs_preprocess_val_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/bc_structured_chunked_validation')
    
    # Stage 3 (CQL) Preprocessed Data
    cql_preprocess_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/cql_preprocessed')
    cql_preprocess_val_dir: str = os.path.expanduser('~/WaymoOfflineAgent/data/cql_preprocessed_validation')

    # Model save paths
    bc_model_path: str = os.path.expanduser('~/WaymoOfflineAgent/models/bc_policy.pth')
    bck_model_path: str = os.path.expanduser('~/WaymoOfflineAgent/models/bc_kinematic_policy.pth')
    cql_model_path: str = os.path.expanduser('~/WaymoOfflineAgent/models/cql_policy.pth')
    bcs_model_path: str = os.path.expanduser('~/WaymoOfflineAgent/models/bc_structured_policy.pth')
    bcs_model_path_v2: str = os.path.expanduser('~/WaymoOfflineAgent/models/bc_structured_policy_v2.pth')
    bcs_t_model_path: str = os.path.expanduser('~/WaymoOfflineAgent/models/bc_structured_transformer_policy.pth')
    
    state_stats_structured_path: str = os.path.expanduser('~/WaymoOfflineAgent/models/state_stats_structured.pt')

    
    # ==========================================================================
    # --- 2. Feature Engineering & Action Space Configurations ---
    # ==========================================================================
    num_closest_agents: int = 15
    num_closest_map_points: int = 50
    num_closest_crosswalk_points: int = 10
    num_future_waypoints: int = 10 # For the 'route' feature
    max_dist: float = 100.0 # Maximum distance for nearest neighbors
    
    action_dim: int = 2 # [acceleration, steering]
    
    # Action clipping limits for preprocessing
    MAX_ACCELERATION: float = 8.0
    MIN_ACCELERATION: float = -10.0
    MAX_STEERING_ANGLE: float = 0.8
    
    # ==========================================================================
    # --- 3. General Training Hyperparameters ---
    # ==========================================================================
    learning_rate: float = 3e-5   # A smaller LR is often better for RL and Transformers
    batch_size: int = 1024       # Final sample batch size
    num_epochs: int = 5000
    num_workers: int = 12
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device: str = 'cpu'
    
    # ==========================================================================
    # --- 4. Stage 3: Offline RL (CQL) Specific Hyperparameters ---
    # ==========================================================================
    gamma: float = 0.95                # Discount factor for future rewards
    tau: float = 0.005                 # Soft update rate for target networks
    
    cql_alpha: float = 10.0              # The conservative penalty weight (the main CQL knob)
    cql_n_actions: int = 10              # Number of "what-if" actions to sample for CQL
    
    tune_alpha: bool = True              # Whether to automatically tune the SAC temperature
    target_entropy: float | None = None  # Target entropy for the policy (if None, defaults to -action_dim)
    
    
    
# It takes a loaded npz 'data' object and returns a SimulatorState
def construct_state_from_npz(data) -> datatypes.SimulatorState:
    """
    Takes a loaded .npz data object and constructs a Waymax SimulatorState.
    This function encapsulates the complex data mapping logic.
    """
    print("Constructing Waymax SimulatorState from .npz data...")
    num_agents = data['object_ids'].shape[0]

    # --- Trajectory ---
    timestamps_seconds = data['timestamps_seconds']
    timestamps_micros_tiled = jnp.tile(timestamps_seconds[None, :], (num_agents, 1)) * 1_000_000
    
    log_trajectory = datatypes.Trajectory(
        x=jnp.array(data['all_trajectories'][:, :, 0]), y=jnp.array(data['all_trajectories'][:, :, 1]),
        z=jnp.array(data['all_trajectories'][:, :, 2]), length=jnp.array(data['all_trajectories'][:, :, 3]),
        width=jnp.array(data['all_trajectories'][:, :, 4]), height=jnp.array(data['all_trajectories'][:, :, 5]),
        yaw=jnp.array(data['all_trajectories'][:, :, 6]), vel_x=jnp.array(data['all_trajectories'][:, :, 7]),
        vel_y=jnp.array(data['all_trajectories'][:, :, 8]), valid=jnp.array(data['all_trajectories'][:, :, 9], dtype=bool),
        timestamp_micros=jnp.array(timestamps_micros_tiled, dtype=jnp.int64),
    )

    # --- Object Metadata ---
    sdc_track_index = int(data['sdc_track_index'])
    is_sdc_mask = jnp.zeros(num_agents, dtype=bool).at[sdc_track_index].set(True)
    object_metadata = datatypes.ObjectMetadata(
        ids=jnp.array(data['object_ids']),
        object_types=jnp.array(data['object_types']),
        is_sdc=is_sdc_mask,
        is_modeled=jnp.array(data['agent_difficulty'] > 0, dtype=bool),
        is_valid=jnp.ones(num_agents, dtype=bool),
        objects_of_interest=jnp.zeros(num_agents, dtype=bool),
        is_controlled=jnp.zeros(num_agents, dtype=bool),
    )

    # --- Roadgraph Points ---
    print("Reconstructing rich RoadgraphPoints object...")

    map_data_dict = data['map_data'].item()
    # Define the Waymax Internal MapElementIds manually.
    # These values are from waymax.datatypes.MapElementIds and are what the visualizer expects.
    WAYMAX_LANE_FREEWAY = 1
    WAYMAX_LANE_SURFACE_STREET = 2
    WAYMAX_ROAD_LINE_UNKNOWN = 5
    WAYMAX_STOP_SIGN = 17
    WAYMAX_CROSSWALK = 18
    WAYMAX_ROAD_EDGE_UNKNOWN = 14
    
    # Initialize empty lists to hold the concatenated data for all points
    points_x, points_y, points_z = [], [], []
    points_ids, points_types = [], []

    # NOTE: For simplicity, we are mapping all lane types to a single Waymax type,
    # and all road line/edge types to a single Waymax type.
    # A more advanced version could map the specific sub-types if needed.

    # Process Lanes
    for feature_id, polyline in map_data_dict['lane_polylines'].items():
        num_points = polyline.shape[0]
        points_x.append(polyline[:, 0])
        points_y.append(polyline[:, 1])
        points_z.append(polyline[:, 2])
        points_ids.extend([feature_id] * num_points)
        points_types.extend([WAYMAX_LANE_SURFACE_STREET] * num_points) # Use a common Waymax lane type

    # Process Road Lines
    for feature_id, polyline in map_data_dict['road_line_polylines'].items():
        num_points = polyline.shape[0]
        points_x.append(polyline[:, 0])
        points_y.append(polyline[:, 1])
        points_z.append(polyline[:, 2])
        points_ids.extend([feature_id] * num_points)
        points_types.extend([WAYMAX_ROAD_LINE_UNKNOWN] * num_points) # Use a common Waymax line type

    # Process Road Edges
    for feature_id, polyline in map_data_dict['road_edge_polylines'].items():
        num_points = polyline.shape[0]
        points_x.append(polyline[:, 0])
        points_y.append(polyline[:, 1])
        points_z.append(polyline[:, 2])
        points_ids.extend([feature_id] * num_points)
        points_types.extend([WAYMAX_ROAD_EDGE_UNKNOWN] * num_points) # Use a common Waymax edge type

    # Process Stop Signs
    for feature_id, stopsign_data in map_data_dict['stopsign_positions'].items():
        pos = stopsign_data['position']
        points_x.append(np.array([pos[0]]))
        points_y.append(np.array([pos[1]]))
        points_z.append(np.array([pos[2]]))
        points_ids.append(feature_id)
        points_types.append(WAYMAX_STOP_SIGN)
        
    # Process Crosswalks
    for feature_id, polygon in map_data_dict['crosswalk_polygons'].items():
        num_points = polygon.shape[0]
        points_x.append(polygon[:, 0])
        points_y.append(polygon[:, 1])
        points_z.append(polygon[:, 2])
        points_ids.extend([feature_id] * num_points)
        points_types.extend([WAYMAX_CROSSWALK] * num_points)

    # Concatenate all lists into final NumPy arrays
    if points_x:
        final_points_x = np.concatenate(points_x)
        final_points_y = np.concatenate(points_y)
        final_points_z = np.concatenate(points_z)
        final_ids = np.array(points_ids, dtype=np.int32)
        final_types = np.array(points_types, dtype=np.int32)
        num_rg_points = len(final_points_x)
    else:
        final_points_x, final_points_y, final_points_z, final_ids, final_types = [], [], [], [], []
        num_rg_points = 0
        
    # Now, create the RoadgraphPoints object with the rich, Waymax-compatible types
    roadgraph_points = datatypes.RoadgraphPoints(
        x=jnp.array(final_points_x),
        y=jnp.array(final_points_y),
        z=jnp.array(final_points_z),
        dir_x=jnp.zeros(num_rg_points, dtype=jnp.float32),
        dir_y=jnp.zeros(num_rg_points, dtype=jnp.float32),
        dir_z=jnp.zeros(num_rg_points, dtype=jnp.float32),
        valid=jnp.ones(num_rg_points, dtype=bool),
        ids=jnp.array(final_ids),
        types=jnp.array(final_types), # This now contains the correct Waymax enums
    )
    # --- NEW: Traffic Light Construction ---
    # Now we construct the real traffic light data, not an empty placeholder.
    tl_states_data = data['traffic_light_states'] # Shape: (91, num_tl_lanes, 4)
    num_timesteps, num_tl_lanes, _ = tl_states_data.shape
    
    # We need to reshape the data to match the TrafficLights constructor
    # The constructor expects separate x,y,z,state,valid fields. Our parser combined some of these.
    # For now, let's just populate the state and valid fields.
    traffic_lights = datatypes.TrafficLights(
        x=jnp.zeros((num_tl_lanes, num_timesteps)), # Placeholder
        y=jnp.zeros((num_tl_lanes, num_timesteps)), # Placeholder
        z=jnp.zeros((num_tl_lanes, num_timesteps)), # Placeholder
        state=jnp.array(tl_states_data[:, :, 1].T, dtype=jnp.int32), # Transpose to get (num_tl_lanes, 91)
        valid=jnp.array(tl_states_data[:, :, 1] > 0, dtype=bool).T, # Valid if state is not UNKNOWN (0). Transpose.
        lane_ids=jnp.array(tl_states_data[0, :, 0], dtype=jnp.int32)[:, None] # Get lane IDs from first step
    )

    # --- Final Assembly ---
    scenario = datatypes.SimulatorState(
        log_trajectory=log_trajectory,
        sim_trajectory=log_trajectory,
        object_metadata=object_metadata,
        roadgraph_points=roadgraph_points,
        log_traffic_light=traffic_lights,
        sdc_paths=None,
        timestep=jnp.array(10, dtype=jnp.int32),
    )
    print("...Construction successful!")
    return scenario


def state_to_feature_vector(state: datatypes.SimulatorState, config: TrainingConfig) -> np.ndarray:
    """
    Converts a live Waymax SimulatorState object into the specific feature
    vector our BC model was trained on. This must EXACTLY match the logic
    in preprocess_for_train_bc_v3.py.
    """
    # Get the current timestep as an integer
    ts = state.timestep.item() 
    
    # Find the SDC's index in the agent list
    sdc_index = jnp.argmax(state.object_metadata.is_sdc).item()

    # --- 1. Establish the Frame of Reference (CORRECTED) ---
    
    # Access each attribute of the sim_trajectory individually
    ego_pos = jnp.array([
        state.sim_trajectory.x[sdc_index, ts],
        state.sim_trajectory.y[sdc_index, ts]
    ])
    ego_heading = state.sim_trajectory.yaw[sdc_index, ts].item()
    
    c, s = np.cos(ego_heading), np.sin(ego_heading)
    rot_matrix = np.array([[c, s], [-s, c]])

    # --- 2. Calculate Ego Features (CORRECTED) ---
    
    current_vel_xy = jnp.array([
        state.sim_trajectory.vel_x[sdc_index, ts],
        state.sim_trajectory.vel_y[sdc_index, ts]
    ])
    current_speed = jnp.linalg.norm(current_vel_xy).item()
    
    if ts > 0:
        prev_vel_xy = jnp.array([
            state.sim_trajectory.vel_x[sdc_index, ts - 1],
            state.sim_trajectory.vel_y[sdc_index, ts - 1]
        ])
        prev_speed = jnp.linalg.norm(prev_vel_xy).item()
        prev_heading = state.sim_trajectory.yaw[sdc_index, ts - 1].item()
        
        acceleration = (current_speed - prev_speed) / 0.1
        yaw_rate = normalize_angle(ego_heading - prev_heading) / 0.1
    else:
        acceleration, yaw_rate = 0.0, 0.0
        
    ego_features = np.array([current_speed, acceleration, yaw_rate], dtype=np.float32)

    # --- 3. Calculate Agent Features (CORRECTED) ---
    all_agents_current_x = state.sim_trajectory.x[:, ts]
    all_agents_current_y = state.sim_trajectory.y[:, ts]
    all_agents_current_valid = state.sim_trajectory.valid[:, ts]

    valid_agents_info = []
    for i in range(state.num_objects):
        if all_agents_current_valid[i] and i != sdc_index:
            agent_pos = jnp.array([all_agents_current_x[i], all_agents_current_y[i]])
            dist = jnp.linalg.norm(agent_pos - ego_pos).item()
            valid_agents_info.append((dist, i))
            
    valid_agents_info.sort(key=lambda item: item[0])
    
    all_agent_features_padded = np.zeros((config.num_closest_agents, 10), dtype=np.float32)
    num_agents_to_add = min(len(valid_agents_info), config.num_closest_agents)

    for i in range(num_agents_to_add):
        _, agent_idx = valid_agents_info[i]
        
        # Get all state components for this agent at the current timestep
        agent_pos_global = jnp.array([state.sim_trajectory.x[agent_idx, ts], state.sim_trajectory.y[agent_idx, ts]])
        agent_vel_global = jnp.array([state.sim_trajectory.vel_x[agent_idx, ts], state.sim_trajectory.vel_y[agent_idx, ts]])
        agent_heading_global = state.sim_trajectory.yaw[agent_idx, ts].item()
        agent_length = state.sim_trajectory.length[agent_idx, ts].item()
        agent_width = state.sim_trajectory.width[agent_idx, ts].item()
        obj_type = state.object_metadata.object_types[agent_idx].item()

        # Perform transformations
        relative_pos = np.dot(rot_matrix, agent_pos_global - ego_pos)
        relative_vel = np.dot(rot_matrix, agent_vel_global)
        relative_heading = normalize_angle(agent_heading_global - ego_heading)
        
        type_vec = np.zeros(3, dtype=np.float32)
        if obj_type in [1, 2, 3]:
            type_vec[obj_type - 1] = 1.0
        
        features = np.concatenate([relative_pos, relative_vel, [relative_heading, agent_length, agent_width], type_vec])
        all_agent_features_padded[i, :] = features
        
    all_agent_features = all_agent_features_padded.flatten()
    
    # 3. Map Features & 4. Rule-Based Features
    
    # --- Lanes & Crosswalks ---
    relative_lane_points = np.zeros((config.num_closest_map_points, 2), dtype=np.float32)
    relative_crosswalk_points = np.zeros((config.num_closest_crosswalk_points, 2), dtype=np.float32)
    
    # We need to reconstruct the point clouds from the RoadgraphPoints object
    rg_points = state.roadgraph_points
    
    # These are the Waymax Internal Enums we discovered
    WAYMAX_LANE_TYPES = {1, 2, 3} # FREEWAY, SURFACE_STREET, BIKE_LANE
    WAYMAX_CROSSWALK_TYPE = 18
    WAYMAX_STOP_SIGN_TYPE = 17
    
    # Filter points by type
    lane_mask = jnp.isin(rg_points.types, jnp.array(list(WAYMAX_LANE_TYPES)))
    all_lane_points = jnp.stack([rg_points.x[lane_mask], rg_points.y[lane_mask]], axis=-1)
    
    cw_mask = (rg_points.types == WAYMAX_CROSSWALK_TYPE)
    all_crosswalk_points = jnp.stack([rg_points.x[cw_mask], rg_points.y[cw_mask]], axis=-1)

    # Perform nearest-neighbor search
    if all_lane_points.shape[0] > 0:
        distances = jnp.linalg.norm(all_lane_points - ego_pos, axis=1)
        closest_indices = jnp.argsort(distances)[:config.num_closest_map_points]
        closest_points_global = all_lane_points[closest_indices]
        transformed_points = np.dot(rot_matrix, (closest_points_global - ego_pos).T).T
        relative_lane_points[:transformed_points.shape[0], :] = transformed_points

    if all_crosswalk_points.shape[0] > 0:
        distances = jnp.linalg.norm(all_crosswalk_points - ego_pos, axis=1)
        closest_indices = jnp.argsort(distances)[:config.num_closest_crosswalk_points]
        closest_points_global = all_crosswalk_points[closest_indices]
        transformed_points = np.dot(rot_matrix, (closest_points_global - ego_pos).T).T
        relative_crosswalk_points[:transformed_points.shape[0], :] = transformed_points

    all_map_features = np.concatenate([relative_lane_points.flatten(), relative_crosswalk_points.flatten()])

    # --- Stop Signs & Traffic Lights (Simplified for now) ---
    dist_to_stop_sign = config.max_dist
    stop_sign_mask = (rg_points.types == WAYMAX_STOP_SIGN_TYPE)
    all_stop_sign_points = jnp.stack([rg_points.x[stop_sign_mask], rg_points.y[stop_sign_mask]], axis=-1)
    if all_stop_sign_points.shape[0] > 0:
        distances = jnp.linalg.norm(all_stop_sign_points - ego_pos, axis=1)
        dist_to_stop_sign = jnp.min(distances).item()

    is_stop_controlled = 0.0 # Placeholder
    tl_state_vec = np.array([0, 0, 0, 1.0], dtype=np.float32) # Placeholder
    
    all_rules_features = np.concatenate([np.array([dist_to_stop_sign, is_stop_controlled]), tl_state_vec])
    
    # Final state vector concatenation
    state_vector = np.concatenate([ego_features, all_agent_features, all_map_features, all_rules_features])
    
    # Final check to ensure consistency
    if state_vector.shape != (279,):
        raise ValueError(f"FATAL: state_to_feature_vector produced shape {state_vector.shape}, expected (279,)")
        
    return state_vector

class WeightedMSELoss(nn.Module):
    """
    A custom Mean Squared Error loss function that applies a weight to each
    sample based on the rarity of its true action.
    """
    def __init__(self, weights_path):
        super().__init__()
        print(f"Loading action weights from: {weights_path}")
        weight_data = torch.load(weights_path, weights_only=True)
        self.weights = weight_data['weights'].requires_grad_(False)
        self.accel_bins = weight_data['accel_bins'].requires_grad_(False)
        self.steer_bins = weight_data['steer_bins'].requires_grad_(False)

    def to(self, device):
        """Moves the weight tensors to the specified device."""
        self.weights = self.weights.to(device)
        self.accel_bins = self.accel_bins.to(device)
        self.steer_bins = self.steer_bins.to(device)
        return self

    def forward(self, pred_actions, true_actions):
        # --- FIX: Make tensor slices contiguous before using them ---
        # This prevents an extra data copy and silences the UserWarning.
        true_accel = true_actions[:, 0].contiguous()
        true_steer = true_actions[:, 1].contiguous()
        
        # Determine the bin index for each true action in the batch
        # Now we pass the contiguous tensors to torch.bucketize
        accel_indices = torch.bucketize(true_accel, self.accel_bins) - 1
        steer_indices = torch.bucketize(true_steer, self.steer_bins) - 1
        
        # Clamp indices to be within the valid range of the weights tensor
        accel_indices = torch.clamp(accel_indices, 0, self.weights.shape[0] - 1)
        steer_indices = torch.clamp(steer_indices, 0, self.weights.shape[1] - 1)
        
        # Look up the weight for each action in the batch
        batch_weights = self.weights[accel_indices, steer_indices]
        
        # Calculate the standard squared error
        squared_errors = (pred_actions - true_actions) ** 2
        
        # Apply the weights. The unsqueeze(1) correctly broadcasts the
        # (batch_size,) weight tensor to the (batch_size, 2) error tensor.
        weighted_squared_errors = batch_weights.unsqueeze(1) * squared_errors
        
        # Return the mean of the weighted errors
        return torch.mean(weighted_squared_errors)