# ======================================================================================
# validate_pipeline.py (Enhanced Version)
#
# Description:
#   This script validates the entire data pipeline and environment setup. It performs
#   the following critical checks:
#   1. Loads a specific processed .npz file (configurable via command-line).
#   2. Prints a summary of the scenario's contents.
#   3. Manually constructs a complete Waymax SimulatorState object, including traffic lights.
#   4. Uses waymax.visualization to plot the constructed state, proving data integrity.
#   5. Uses waymax.env to initialize a simulation, proving compatibility with the core engine.
#
# A successful run of this script confirms that Stage 1 is complete and the
# environment is ready for model development in Stage 2.
#
# Author: Antonio Guillen-Perez
# Date: 2025-07-29
# ======================================================================================

import numpy as np
import jax
from jax import numpy as jnp
import os
import glob
import argparse  # NEW: For command-line arguments

# Import the Waymax libraries
from waymax import datatypes
from waymax import visualization
from waymax import env as _env
from waymax import dynamics
from waymax import config as _config

from src.utils import construct_state_from_npz

# --- Configuration ---
PROCESSED_DATA_DIR = os.path.expanduser('~/WaymoOfflineAgent/data/processed_validation')
OUTPUT_DIR = os.path.expanduser('~/WaymoOfflineAgent/outputs/validation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main(args):
    print("--- Starting Pipeline Validation ---")
    
    # --- 1. Load a Processed .npz File ---
    if args.scenario_id:
        filepath = os.path.join(PROCESSED_DATA_DIR, f"{args.scenario_id}.npz")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Specified scenario ID not found: {filepath}")
    else:
        npz_files = glob.glob(os.path.join(PROCESSED_DATA_DIR, '*.npz'))
        if not npz_files:
            raise FileNotFoundError(f"No processed .npz files found in {PROCESSED_DATA_DIR}. Please run parser.py first.")
        filepath = np.random.choice(npz_files) # Load a random scenario
    
    print(f"Loading data from: {os.path.basename(filepath)}")
    data = np.load(filepath, allow_pickle=True)

    # --- NEW: Print Scenario Summary ---
    num_agents = data['object_ids'].shape[0]
    object_types = data['object_types']
    num_vehicles = np.sum(object_types == 1)
    num_pedestrians = np.sum(object_types == 2)
    num_cyclists = np.sum(object_types == 3)
    other_agents = np.sum(object_types == 4)  
    map_data_dict = data['map_data'].item()
    print("\n--- Scenario Summary ---")
    print(f"  Scenario ID: {data['scenario_id']}")
    print(f"  Total Agents: {num_agents} (Vehicles: {num_vehicles}, Pedestrians: {num_pedestrians}, Cyclists: {num_cyclists}, Other: {other_agents})")
    print(f"  Map Features: {len(map_data_dict['lane_polylines'])} Lanes, {len(map_data_dict['road_line_polylines'])} Road Lines, {len(map_data_dict['crosswalk_polygons'])} Crosswalks")
    print("------------------------\n")

    # --- 2. Manually Construct the Waymax SimulatorState Object ---
    scenario = construct_state_from_npz(data)

    # --- 3. Validate with `visualization` ---
    print("Validating with waymax.visualization...")
    img = visualization.plot_simulator_state(scenario, use_log_traj=True)
    output_image_path = os.path.join(OUTPUT_DIR, f"{data['scenario_id']}_validation.png")
    visualization.utils.save_img_as_png(img, output_image_path)
    print(f"Saved validation image to: {output_image_path}")
    print("...Visualization SUCCESSFUL.")

    # --- 4. Validate with `env` (The Final Test) ---
    print("Validating with waymax.env...")
    env_config = _config.EnvironmentConfig(
        max_num_objects=num_agents,
        controlled_object=_config.ObjectType.SDC,
        metrics=_config.MetricsConfig(),
        rewards=_config.LinearCombinationRewardConfig(rewards={}),
    )
    dynamics_model = dynamics.StateDynamics()
    env = _env.MultiAgentEnvironment(dynamics_model=dynamics_model, config=env_config)
    state = env.reset(scenario)
    
    if state.log_trajectory.valid.any():
        print("...env.reset() SUCCESSFUL.")
    else:
        raise RuntimeError("env.reset() failed to produce a valid state.")

    print("\n--- ✅ All Pipeline Validations Passed! ---")
    print("Stage 1 is complete. The environment is fully configured and ready for Stage 2.")


if __name__ == '__main__':
    # --- NEW: Add Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Validate the Waymax data pipeline.")
    parser.add_argument(
        '--scenario_id',
        type=str,
        default=None,
        help='Specific scenario ID to validate (without the .npz extension). If not provided, a random scenario is chosen.'
    )
    args = parser.parse_args()
    main(args)