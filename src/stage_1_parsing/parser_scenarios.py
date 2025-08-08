# ======================================================================================
# parser_scenarios.py
#
# Description:
#   This script performs a one-time, offline parsing of the Waymo Open Motion Dataset.
#   Source: https://waymo.com/open/
#   It reads the raw, sharded .tfrecord files containing Scenario protos, extracts
#   key information, structures it, and saves it into a series of compressed,
#   efficient NumPy .npz files. Each .npz file corresponds to a single scenario.
#   Use conda activate womd-parser
# Author: Antonio Guillen-Perez
# Date: 2025-07-29
# ======================================================================================

import tensorflow as tf
import numpy as np
import os
import glob
from tqdm import tqdm
import multiprocessing

# Import the official Waymo Open Dataset protocol buffer definitions
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import map_pb2


def process_single_file(file_path_tuple):
    """
    This is the "worker" function. It contains the logic to parse all scenarios
    within a single .tfrecord file and save them as .npz files.
    """
    file_path, output_dir = file_path_tuple
    try:
        dataset = tf.data.TFRecordDataset(file_path, compression_type='')
        for data in dataset:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data.numpy())

            # A. Basic Information
            scenario_id = scenario.scenario_id
            timestamps_seconds = np.array(scenario.timestamps_seconds, dtype=np.float32)
            sdc_track_index = scenario.sdc_track_index

            # B. Agent Data
            num_agents = len(scenario.tracks)
            difficulty_map = {p.track_index: p.difficulty for p in scenario.tracks_to_predict}
            all_trajectories = np.zeros((num_agents, 91, 10), dtype=np.float32)
            object_ids = np.zeros(num_agents, dtype=np.int32)
            object_types = np.zeros(num_agents, dtype=np.int32)
            agent_difficulty = np.zeros(num_agents, dtype=np.int32)

            for i, track in enumerate(scenario.tracks):
                object_ids[i] = track.id
                object_types[i] = track.object_type
                agent_difficulty[i] = difficulty_map.get(i, 0)
                for t, state in enumerate(track.states):
                    all_trajectories[i, t, 0] = state.center_x
                    all_trajectories[i, t, 1] = state.center_y
                    all_trajectories[i, t, 2] = state.center_z
                    all_trajectories[i, t, 3] = state.length
                    all_trajectories[i, t, 4] = state.width
                    all_trajectories[i, t, 5] = state.height
                    all_trajectories[i, t, 6] = state.heading
                    all_trajectories[i, t, 7] = state.velocity_x
                    all_trajectories[i, t, 8] = state.velocity_y
                    all_trajectories[i, t, 9] = float(state.valid)

            # C. Dynamic Map Data (Traffic Lights)
            # This logic needs to be careful about scenarios with no traffic lights
            if scenario.dynamic_map_states:
                tl_lane_to_idx = {tl.lane: i for i, tl in enumerate(scenario.dynamic_map_states[0].lane_states)}
                num_tl_lanes = len(tl_lane_to_idx)
            else:
                num_tl_lanes = 0
            
            traffic_light_states = np.zeros((91, num_tl_lanes, 4), dtype=np.float32)
            if num_tl_lanes > 0:
                for t, dynamic_state in enumerate(scenario.dynamic_map_states):
                    for lane_state in dynamic_state.lane_states:
                        if lane_state.lane in tl_lane_to_idx:
                            idx = tl_lane_to_idx[lane_state.lane]
                            traffic_light_states[t, idx, 0] = lane_state.lane
                            traffic_light_states[t, idx, 1] = lane_state.state
                            if lane_state.HasField('stop_point'):
                                traffic_light_states[t, idx, 2] = lane_state.stop_point.x
                                traffic_light_states[t, idx, 3] = lane_state.stop_point.y

            # D. Static Map Data
            map_data = {
                'lane_polylines': {}, 'road_line_polylines': {}, 'road_edge_polylines': {},
                'crosswalk_polygons': {}, 'stopsign_positions': {}, 'lane_connectivity': {}
            }
            for feature in scenario.map_features:
                feature_id = feature.id
                if feature.HasField('lane'):
                    map_data['lane_polylines'][feature_id] = np.array([[p.x, p.y, p.z] for p in feature.lane.polyline], dtype=np.float32)
                    map_data['lane_connectivity'][feature_id] = {
                        'left_neighbors': [n.feature_id for n in feature.lane.left_neighbors],
                        'right_neighbors': [n.feature_id for n in feature.lane.right_neighbors]
                    }
                elif feature.HasField('road_line'):
                    map_data['road_line_polylines'][feature_id] = np.array([[p.x, p.y, p.z] for p in feature.road_line.polyline], dtype=np.float32)
                elif feature.HasField('road_edge'):
                    map_data['road_edge_polylines'][feature_id] = np.array([[p.x, p.y, p.z] for p in feature.road_edge.polyline], dtype=np.float32)
                elif feature.HasField('crosswalk'):
                    map_data['crosswalk_polygons'][feature_id] = np.array([[p.x, p.y, p.z] for p in feature.crosswalk.polygon], dtype=np.float32)
                elif feature.HasField('stop_sign'):
                    pos = feature.stop_sign.position
                    map_data['stopsign_positions'][feature_id] = {
                        'position': np.array([pos.x, pos.y, pos.z], dtype=np.float32),
                        'controls_lanes': list(feature.stop_sign.lane)
                    }

            # E. Saving the Output
            output_filename = os.path.join(output_dir, f"{scenario_id}.npz")
            np.savez_compressed(
                output_filename,
                scenario_id=scenario_id, timestamps_seconds=timestamps_seconds,
                sdc_track_index=np.int32(sdc_track_index), all_trajectories=all_trajectories,
                object_ids=object_ids, object_types=object_types, agent_difficulty=agent_difficulty,
                traffic_light_states=traffic_light_states, map_data=map_data
            )
    except Exception as e:
        # A file might be corrupted, this prevents the whole process from crashing.
        print(f"WARNING: Failed to process file {os.path.basename(file_path)} due to error: {e}")
        pass


def main():
    """
    Main function to orchestrate the parallel parsing of training and validation sets.
    """
    # Use half of the available CPU cores to avoid overwhelming the system,
    # leaving resources for other tasks. Can be set to cpu_count() for max speed.
    num_workers = multiprocessing.cpu_count() // 2
    print(f"Using {num_workers} CPU cores for parallel processing.")
    
    # Process both the training and validation splits
    for data_split in ['training', 'validation']:
        
        # --- Configure paths for the current split ---
        input_dir = f'/mnt/d/waymo_datasets/uncompressed/scenario/{data_split}/'
        output_dir = os.path.expanduser(f'~/WaymoOfflineAgent/data/processed_{data_split}')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n--- Processing '{data_split}' split ---")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        tfrecord_files = glob.glob(os.path.join(input_dir, '*.tfrecord*'))
        if not tfrecord_files:
            print(f"WARNING: No .tfrecord files found in '{input_dir}'. Skipping.")
            continue
            
        print(f"Found {len(tfrecord_files)} .tfrecord shard files to process.")

        # Prepare a list of tasks for the worker pool. Each task is a tuple.
        tasks = [(file_path, output_dir) for file_path in tfrecord_files]
        
        # --- Create and run the multiprocessing Pool ---
        # `imap_unordered` is efficient for memory and works great with tqdm
        with multiprocessing.Pool(processes=num_workers) as pool:
            # tqdm will wrap the iterator returned by the pool to create the progress bar
            list(tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks), desc=f"Parsing {data_split} shards"))

    print("\n--- All Parsing Complete ---")


if __name__ == '__main__':
    # This check is a best practice for multiprocessing code
    main()