# ======================================================================================
# evaluate_bc_stage_2_5.py
#
# Description:
#   Loads the trained Kinematic Behavioral Cloning (BC-K) policy and evaluates
#   it in a closed-loop simulation using the Waymax environment with the
#   InvertibleBicycleModel for realistic vehicle dynamics.
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
import random

# Import our project-specific modules
# We need the MLP class from the training script and the centralized config
from src.stage_2_5_kinematic_bc.train import MLP 
from src.shared.utils import TrainingConfig, construct_state_from_npz, state_to_feature_vector

# Import Waymax components
from waymax import datatypes
from waymax import visualization
from waymax import env as _env
from waymax import dynamics
from waymax import config as _config

def main(args):
    print("--- Starting Stage 2.5: Kinematic BC Policy Evaluation ---")
    config = TrainingConfig()

    # --- 1. Load Scenario and Environment ---
    print(f"Loading scenario: {args.scenario_id}")
    # We load the raw processed data, not the BC-preprocessed data
    filepath = os.path.join(config.val_data_dir, f"{args.scenario_id}.npz")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scenario file not found: {filepath}. Ensure you are using a valid scenario ID from the validation set.")
    
    data = np.load(filepath, allow_pickle=True)
    
    # Use our utility function to construct the initial Waymax state
    initial_state = construct_state_from_npz(data)
    
    # Create the environment config
    env_config = _config.EnvironmentConfig(
        max_num_objects=initial_state.num_objects,
        controlled_object=_config.ObjectType.SDC,
        metrics=_config.MetricsConfig(),
        rewards=_config.LinearCombinationRewardConfig(rewards={}),
    )
    
    # --- KEY CHANGE: Use the InvertibleBicycleModel ---
    print("Initializing environment with InvertibleBicycleModel...")
    dynamics_model = dynamics.InvertibleBicycleModel()
    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=env_config,
    )

    # --- 2. Load Trained PyTorch Model ---
    print(f"Loading trained kinematic model from: {config.bck_model_path}")
    model = MLP(input_dim=279) # Ensure input dim matches config
    model.load_state_dict(torch.load(config.bck_model_path, map_location=config.device))
    model.to(config.device)
    model.eval() # Set model to evaluation mode

    # --- 3. The Simulation Loop ---
    print("Starting closed-loop simulation...")
    current_state = env.reset(initial_state)
    rollout_states = [current_state]
    
    # Get the SDC index once at the beginning
    sdc_index = jnp.argmax(current_state.object_metadata.is_sdc).item()

    for _ in tqdm(range(current_state.remaining_timesteps), desc="Simulating"):
        # a. Featurize: Convert the live Waymax state to our model's input vector
        feature_vector = state_to_feature_vector(current_state, config)
        
        # b. Infer Action (PyTorch)
        with torch.no_grad():
            state_tensor = torch.from_numpy(feature_vector).to(config.device, non_blocking=True).float().unsqueeze(0)
            action_kinematic_tensor = model(state_tensor)
            action_kinematic = action_kinematic_tensor.squeeze(0).cpu().numpy() # Shape: (2,) -> [accel, steer]

        # c. Step the Environment (Waymax)
        # The InvertibleBicycleModel expects a (num_objects, 2) action array.
        all_agent_actions = jnp.zeros((current_state.num_objects, 2))
        final_action_data = all_agent_actions.at[sdc_index].set(action_kinematic)

        # The validity mask is the same as before
        action_valid_mask = jnp.zeros((current_state.num_objects, 1), dtype=bool)
        final_action_valid = action_valid_mask.at[sdc_index].set(True)

        waymax_action = datatypes.Action(
            data=final_action_data,
            valid=final_action_valid
        )
        
        # JIT-compile the step function for a significant speedup
        jit_step = jax.jit(env.step)
        current_state = jit_step(current_state, waymax_action)
        
        rollout_states.append(current_state)

    print("...Simulation complete.")

    # --- 4. Visualization ---
    print("Generating rollout video...")
    imgs = []
    # We plot the sim_trajectory (use_log_traj=False) to see our agent's actions
    for state in tqdm(rollout_states, desc="Rendering frames"):
        imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))

    output_path = os.path.join(args.output_dir, f"{args.scenario_id}_bck_rollout.mp4")
    
    # Use mediapy to save the video
    mediapy.write_video(output_path, imgs, fps=10)
    print(f"--- ✅ Kinematic Evaluation Complete. Video saved to: {output_path} ---")

if __name__ == "__main__":
    config = TrainingConfig() # Load config to get the data path

    parser = argparse.ArgumentParser(description="Evaluate a trained Kinematic Behavioral Cloning policy (Stage 2.5).")
    parser.add_argument(
        "--scenario_id",
        type=str,
        default=None,  # Default to None, so we can check if the user provided one
        help="Specific scenario ID to evaluate. If not provided, a random scenario is chosen from the validation set."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.expanduser('~/WaymoOfflineAgent/outputs/evaluations/stage_2_5'),
        help="Directory to save the output video."
    )
    args = parser.parse_args()

    # If no scenario ID was provided by the user, pick one randomly.
    if args.scenario_id is None:
        print("No scenario ID provided. Selecting a random scenario from the validation set...")
        # Scan the validation directory for all available scenarios
        validation_files = glob.glob(os.path.join(config.val_data_dir, '*.npz'))
        if not validation_files:
            raise FileNotFoundError(f"No .npz files found in the validation directory: {config.val_data_dir}")
        
        # Select one random file path
        random_filepath = random.choice(validation_files)
        
        # Extract just the scenario ID (the filename without the extension)
        args.scenario_id = os.path.basename(random_filepath).replace('.npz', '')
        print(f"Randomly selected scenario: {args.scenario_id}")

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
