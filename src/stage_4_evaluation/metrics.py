# ======================================================================================
# metrics.py (Stage 4 - Corrected & Final)
#
# Description:
#   This final version uses the direct, explicit properties of the Trajectory
#   dataclass for maximum clarity and correctness.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import jax
from jax import numpy as jnp
import numpy as np

from waymax import datatypes
from waymax.utils.geometry import compute_pairwise_overlaps
from waymax.utils import geometry

# --- Metric Constants (Tunable) ---
COLLISION_IOU_THRESHOLD = 0.5
OFF_ROAD_THRESHOLD = 2.0
GOAL_THRESHOLD = 3.0

def check_collision(
    state: datatypes.SimulatorState,
    sdc_index: int
) -> bool:
    """
    Checks if the SDC has collided with any other valid object during the rollout.
    This definitive version manually constructs the 5-DoF bounding box tensors
    required by the low-level `check_overlap` geometry function.
    """
    sim_traj = state.sim_trajectory
    valid_mask = sim_traj.valid # Shape: (num_objects, num_timesteps)

    # 1. Manually stack the 5 DoF for all agents at all timesteps.
    # The result `all_bboxes_5dof` will have the shape (num_objects, num_timesteps, 5)
    all_bboxes_5dof = sim_traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])

    # 2. Extract the SDC's trajectory of 5-DoF bboxes
    sdc_bboxes_5dof = all_bboxes_5dof[sdc_index] # Shape: (num_timesteps, 5)
    
    # 3. Define a function to check for collisions in a single timestep
    def check_collision_at_timestep(t):
        # Get the SDC's 5-DoF bbox at this timestep
        sdc_bbox = sdc_bboxes_5dof[t] # Shape: (5,)
        
        # Get all other agents' 5-DoF bboxes and validity at this timestep
        other_agents_bboxes = all_bboxes_5dof[:, t, :] # Shape: (num_objects, 5)
        other_agents_valid = valid_mask[:, t]      # Shape: (num_objects,)

        # 4. Use vmap to efficiently check the SDC against every other agent
        #    The `check_overlap` function takes two tensors of shape (..., 5)
        #    and returns a boolean. `vmap` will broadcast the single sdc_bbox
        #    against the array of other_agents_bboxes.
        #    The output `all_overlaps_at_t` will be a boolean array of shape (num_objects,).
        all_overlaps_at_t = jax.vmap(geometry.has_overlap, in_axes=(None, 0))(sdc_bbox, other_agents_bboxes)

        # 5. We only care about collisions with other *valid* agents.
        #    Create a mask that is True for agents that are not the SDC.
        not_sdc_mask = (jnp.arange(state.num_objects) != sdc_index)
        
        # A collision is an overlap with another agent that is BOTH valid AND not the SDC.
        valid_collisions_mask = all_overlaps_at_t & other_agents_valid & not_sdc_mask
        
        # If any of these are true, a collision happened at this timestep.
        return jnp.any(valid_collisions_mask)

    # 6. Use vmap again to run the single-timestep check across all timesteps
    num_timesteps = state.sim_trajectory.num_timesteps
    collisions_over_time = jax.vmap(check_collision_at_timestep)(jnp.arange(num_timesteps))

    # 7. If any timestep had a collision, the whole scenario has a collision.
    return bool(jnp.any(collisions_over_time))

def check_off_road(
    state: datatypes.SimulatorState,
    sdc_index: int
) -> bool:
    """
    Checks if the SDC has driven off the road at any point during the rollout.
    """
    # This function is already correct as it uses the .xy property.
    sdc_positions = state.sim_trajectory.xy[sdc_index]
    
    rg_points = state.roadgraph_points
    # We need to find the correct enum for ROAD_EDGE. A safer way is to check multiple.
    ROAD_EDGE_TYPES = jnp.array([15, 16]) # ROAD_EDGE_BOUNDARY, ROAD_EDGE_MEDIAN
    
    edge_mask = jnp.isin(rg_points.types, ROAD_EDGE_TYPES)
    if not edge_mask.any():
        return False
        
    road_edge_points = jnp.stack([rg_points.x[edge_mask], rg_points.y[edge_mask]], axis=-1)
    
    def min_dist_to_edges(point):
        return jnp.min(jnp.linalg.norm(road_edge_points - point, axis=1))

    min_distances_over_time = jax.vmap(min_dist_to_edges)(sdc_positions)
    
    # We only care about timesteps where the SDC was valid
    sdc_valid_mask = state.sim_trajectory.valid[sdc_index]
    valid_distances = min_distances_over_time[sdc_valid_mask]
    
    if valid_distances.shape[0] == 0:
        return False

    max_dist_from_road = jnp.max(valid_distances)
    
    return bool(max_dist_from_road > OFF_ROAD_THRESHOLD)

def check_goal_completion(
    state: datatypes.SimulatorState,
    sdc_index: int
) -> bool:
    """
    Checks if the SDC successfully reached its destination by the end of the rollout.
    """
    # This function is also correct as it uses the .xy property.
    final_sim_position = state.sim_trajectory.xy[sdc_index, -1]
    final_log_position = state.log_trajectory.xy[sdc_index, -1]
    is_final_state_valid = state.sim_trajectory.valid[sdc_index, -1]
    
    distance_to_goal = jnp.linalg.norm(final_sim_position - final_log_position)
    
    return bool((is_final_state_valid) & (distance_to_goal < GOAL_THRESHOLD))