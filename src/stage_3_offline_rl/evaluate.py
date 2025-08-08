# ======================================================================================
# evaluate.py (Stage 3 - CQL)
#
# Description:
#   Loads a trained CQL policy (specifically, the Actor network) and evaluates
#   it in a closed-loop simulation using the Waymax environment.
#
# Author: Antonio Guillen-Perez
# ======================================================================================

import torch
import numpy as np
import jax
from jax import numpy as jnp
import os
import glob
from tqdm import tqdm
import argparse
import mediapy
import re # NEW: Import the regular expression module

# Import our project-specific modules
from src.shared.utils import TrainingConfig, construct_state_from_npz, state_to_feature_vector
from src.stage_3_offline_rl.networks import Actor # We only need the Actor for evaluation

# Import Waymax components
from waymax import datatypes
from waymax import visualization
from waymax import env as _env
from waymax import dynamics
from waymax import config as _config

def main(args):
    print("--- Starting Stage 3: CQL Policy Evaluation ---")
    config = TrainingConfig()

    # --- 1. Load Scenario and Environment ---
    print(f"Loading scenario: {args.scenario_id}")
    filepath = os.path.join(config.val_data_dir, f"{args.scenario_id}.npz")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scenario file not found: {filepath}.")
    
    data = np.load(filepath, allow_pickle=True)
    initial_state = construct_state_from_npz(data)
    
    env_config = _config.EnvironmentConfig(
        max_num_objects=initial_state.num_objects,
        controlled_object=_config.ObjectType.SDC,
        metrics=_config.MetricsConfig(),
        rewards=_config.LinearCombinationRewardConfig(rewards={}),
    )
    dynamics_model = dynamics.InvertibleBicycleModel()
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=env_config,
    )

    # --- 2. Load State Normalization Statistics ---
    print("Loading state normalization statistics...")
    stats_path = os.path.join(os.path.dirname(config.cql_model_path), 'state_stats.pt')
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"State stats not found at {stats_path}. Please run compute_state_stats.py first.")
    
    stats = torch.load(stats_path, weights_only=True)
    state_mean = stats['mean'].numpy()
    state_std = stats['std'].numpy()
    
    # --- 3. Load Trained Actor Network ---
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else config.cql_model_path
    print(f"Loading trained actor from checkpoint: {checkpoint_path}")
    
    # Use a regular expression to find the run ID and epoch number from the path
    # This pattern looks for '/cql_.../models/cql_policy_epoch_XX.pth'
    match = re.search(r'/(cql_[^/]+)/models/cql_policy_epoch_(\d+)\.pth', checkpoint_path)
    
    if match:
        run_id = match.group(1)
        epoch_num = match.group(2)
        video_filename = f"{args.scenario_id}__{run_id}__epoch_{epoch_num}.mp4"
    else:
        # Fallback for a generic or non-standard checkpoint path
        video_filename = f"{args.scenario_id}_cql_rollout.mp4"
        
    
    actor = Actor(state_dim=279, action_dim=2)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.to(config.device)
    actor.eval() # Set model to evaluation mode

    # --- 4. The Simulation Loop ---
    print("Starting closed-loop simulation...")
    current_state = env.reset(initial_state)
    rollout_states = [current_state]
    sdc_index = jnp.argmax(current_state.object_metadata.is_sdc).item()
    jit_step = jax.jit(env.step)

    for _ in tqdm(range(current_state.remaining_timesteps), desc="Simulating"):
        # a. Featurize the current state
        feature_vector = state_to_feature_vector(current_state, config)
        
        # b. Normalize the state vector
        normalized_feature_vector = (feature_vector - state_mean) / state_std
        
        # c. Infer Action (PyTorch)
        with torch.no_grad():
            state_tensor = torch.from_numpy(normalized_feature_vector).to(config.device).float().unsqueeze(0)
            
            # For evaluation, we take the deterministic action (the mean of the distribution)
            mean, _ = actor(state_tensor)
            
            # The actor outputs an action in the range [-1, 1] due to tanh. We need to rescale it.
            # We use the actor's own scaling factors to do this correctly.
            action_rescaled = mean * actor.action_scale + actor.action_bias
            
            action_kinematic = action_rescaled.squeeze(0).cpu().numpy()

        # d. Step the Environment (Waymax)
        all_agent_actions = jnp.zeros((current_state.num_objects, 2))
        final_action_data = all_agent_actions.at[sdc_index].set(action_kinematic)
        
        # The validity mask is the same as before
        action_valid_mask = jnp.zeros((current_state.num_objects, 1), dtype=bool).at[sdc_index].set(True)
        final_action_valid = action_valid_mask.at[sdc_index].set(True)

        waymax_action = datatypes.Action(data=final_action_data, valid=final_action_valid)
        
        # JIT-compile the step function for a significant speedup
        jit_step = jax.jit(env.step)
        
        current_state = jit_step(current_state, waymax_action)
        rollout_states.append(current_state)

    print("...Simulation complete.")

    # --- 5. Visualization ---
    print("Generating rollout video...")
    imgs = []
    for state in tqdm(rollout_states, desc="Rendering frames"):
        imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))

    # output_path = os.path.join(args.output_dir, f"{args.scenario_id}_cql_rollout.mp4")
    output_path = os.path.join(args.output_dir, video_filename)

    mediapy.write_video(output_path, imgs, fps=10)
    print(f"--- ✅ CQL Evaluation Complete. Video saved to: {output_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained CQL policy (Stage 3).")
    parser.add_argument(
        "scenario_id", type=str,
        help="The scenario ID from the validation set to evaluate."
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Path to a specific model checkpoint .pth file. If not provided, uses the path from TrainingConfig."
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.expanduser('~/WaymoOfflineAgent/outputs/evaluations/state_3'),
        help="Directory to save the output video."
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)