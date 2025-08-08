#%%
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib import animation
from IPython.display import HTML
from tqdm import tqdm

# Important: We need the official Waymo 'Scenario' protobuf definition
# to be able to decode the data.
from waymo_open_dataset.protos import scenario_pb2, map_pb2

# --- Configuration ---
# Let's point to the VERY FIRST shard file on your D: drive.
TFRECORD_FILE = '/mnt/d/waymo_datasets/uncompressed/scenario/validation/uncompressed_scenario_validation_validation.tfrecord-00000-of-00150'

# How many scenarios from the start of the file do we want to look at?
NUM_SCENARIOS_TO_INSPECT = 9

# --- Styling Dictionaries ---
# Create dictionaries to map enums to plot styles for a much richer map
ROAD_LINE_TYPE_STYLE = {
    map_pb2.RoadLine.TYPE_BROKEN_SINGLE_WHITE: {'color': 'white', 'linestyle': '--', 'linewidth': 1.0},
    map_pb2.RoadLine.TYPE_SOLID_SINGLE_WHITE: {'color': 'white', 'linestyle': '-', 'linewidth': 1.5},
    map_pb2.RoadLine.TYPE_SOLID_DOUBLE_WHITE: {'color': 'white', 'linestyle': '-', 'linewidth': 2.0},
    map_pb2.RoadLine.TYPE_BROKEN_SINGLE_YELLOW: {'color': 'yellow', 'linestyle': '--', 'linewidth': 1.0},
    map_pb2.RoadLine.TYPE_BROKEN_DOUBLE_YELLOW: {'color': 'yellow', 'linestyle': '--', 'linewidth': 2.0},
    map_pb2.RoadLine.TYPE_SOLID_SINGLE_YELLOW: {'color': 'yellow', 'linestyle': '-', 'linewidth': 1.5},
    map_pb2.RoadLine.TYPE_SOLID_DOUBLE_YELLOW: {'color': 'yellow', 'linestyle': '-', 'linewidth': 2.0},
    map_pb2.RoadLine.TYPE_PASSING_DOUBLE_YELLOW: {'color': 'yellow', 'linestyle': '-.', 'linewidth': 2.0},
}

TRAFFIC_LIGHT_STYLE = {
    map_pb2.TrafficSignalLaneState.LANE_STATE_STOP: 'red',
    map_pb2.TrafficSignalLaneState.LANE_STATE_CAUTION: 'yellow',
    map_pb2.TrafficSignalLaneState.LANE_STATE_GO: 'green',
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_STOP: 'red',
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_CAUTION: 'yellow',
    map_pb2.TrafficSignalLaneState.LANE_STATE_ARROW_GO: 'green',
    map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_STOP: 'red',
    map_pb2.TrafficSignalLaneState.LANE_STATE_FLASHING_CAUTION: 'yellow',
}

def plot_timestep(scenario, timestep_index, ax, show_legend=False):
    """
    Generates a rich, top-down plot of a single timestep on a GIVEN axis object.

    Args:
        scenario: The parsed scenario_pb2.Scenario object.
        timestep_index: The integer index of the timestep to plot.
        ax: The matplotlib axes object to draw on.
        show_legend: If True, a legend will be drawn on the plot.
    """
    ax.clear() # Clear previous drawings
    ax.set_facecolor('0.2')

    # Get object groups for highlighting
    tracks_to_predict_indices = {p.track_index for p in scenario.tracks_to_predict}

    # Plot Static Map Features
    for feature in scenario.map_features:
        if feature.HasField('lane'):
            points = np.array([[p.x, p.y] for p in feature.lane.polyline])
            ax.plot(points[:, 0], points[:, 1], color='grey', linestyle='--', linewidth=0.5, zorder=1)
        elif feature.HasField('road_line'):
            style = ROAD_LINE_TYPE_STYLE.get(feature.road_line.type, {})
            points = np.array([[p.x, p.y] for p in feature.road_line.polyline])
            ax.plot(points[:, 0], points[:, 1], **style, zorder=2)
        elif feature.HasField('road_edge'):
            points = np.array([[p.x, p.y] for p in feature.road_edge.polyline])
            ax.plot(points[:, 0], points[:, 1], color='dimgray', linewidth=2, zorder=2)
        elif feature.HasField('stop_sign'):
            pos = feature.stop_sign.position
            ax.scatter(pos.x, pos.y, color='red', marker='8', s=150, zorder=10)
        elif feature.HasField('crosswalk'):
            points = np.array([[p.x, p.y] for p in feature.crosswalk.polygon])
            ax.add_patch(Polygon(points, facecolor='blue', alpha=0.2, zorder=1))

    # Plot Dynamic Map Features (Traffic Lights)
    if timestep_index < len(scenario.dynamic_map_states):
        dynamic_state = scenario.dynamic_map_states[timestep_index]
        for lane_state in dynamic_state.lane_states:
            color = TRAFFIC_LIGHT_STYLE.get(lane_state.state, 'white')
            if lane_state.HasField('stop_point'):
                sp = lane_state.stop_point
                ax.plot(sp.x, sp.y, 'o', color=color, markersize=10, zorder=11)

    # Plot all agents and their trajectories
    agent_centers_x, agent_centers_y = [], []
    for i, track in enumerate(scenario.tracks):
        state = track.states[timestep_index]
        if not state.valid: continue
        agent_centers_x.append(state.center_x)
        agent_centers_y.append(state.center_y)
        face_color, edge_color, zorder = 'red', 'black', 15
        is_special_agent = False
        if i == scenario.sdc_track_index:
            face_color, edge_color, zorder = 'blue', 'cyan', 20
            is_special_agent = True
        elif i in tracks_to_predict_indices:
            face_color, edge_color, zorder = 'orange', 'yellow', 18
            is_special_agent = True
        if is_special_agent:
            past_points = np.array([[s.center_x, s.center_y] for s in track.states[:timestep_index] if s.valid])
            future_points = np.array([[s.center_x, s.center_y] for s in track.states[timestep_index:] if s.valid])
            if past_points.size > 0: ax.plot(past_points[:, 0], past_points[:, 1], color=face_color, linewidth=1.5, alpha=0.7, zorder=14)
            if future_points.size > 0: ax.plot(future_points[:, 0], future_points[:, 1], color=face_color, linestyle='--', linewidth=1.5, alpha=0.7, zorder=14)
        l, w = state.length, state.width
        corners = np.array([[-l/2,-w/2],[-l/2,w/2],[l/2,w/2],[l/2,-w/2]])
        rot_mat = np.array([[np.cos(state.heading), -np.sin(state.heading)], [np.sin(state.heading), np.cos(state.heading)]])
        final_corners = np.dot(corners, rot_mat.T) + np.array([state.center_x, state.center_y])
        ax.add_patch(Polygon(final_corners, facecolor=face_color, edgecolor=edge_color, zorder=zorder, alpha=0.9, linewidth=1.5))

    # Finalize plot appearance
    if agent_centers_x:
        center_x, center_y = np.mean(agent_centers_x), np.mean(agent_centers_y)
        ax.set_xlim(center_x - 75, center_x + 75)
        ax.set_ylim(center_y - 75, center_y + 75)
    ax.set_title(f"Scenario {scenario.scenario_id[:8]} at Timestep {timestep_index}", color='white', fontsize=16)
    ax.set_xlabel("X coordinate (meters)", color='white')
    ax.set_ylabel("Y coordinate (meters)", color='white')
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # --- ADD LEGEND IF REQUESTED ---
    if show_legend:
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', label='SDC', markerfacecolor='blue', markeredgecolor='cyan', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='s', color='w', label='Track to Predict', markerfacecolor='orange', markeredgecolor='yellow', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='s', color='w', label='Other Agent', markerfacecolor='red', markeredgecolor='black', markersize=10, linestyle='None'),
            Line2D([0], [0], color='yellow', linestyle='-', label='Solid Yellow Line', linewidth=2),
            Line2D([0], [0], color='white', linestyle='--', label='Broken White Line', linewidth=2),
            Line2D([0], [0], marker='o', color='w', label='Traffic Light (Green)', markerfacecolor='green', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='8', color='red', label='Stop Sign', markersize=12, linestyle='None')
        ]
        ax.legend(handles=legend_elements, loc='upper right', facecolor='black', labelcolor='white', fontsize='small')


def create_scenario_video(scenario, output_folder):
    """
    Generates and saves a video of the full 9-second scenario.
    """
    print(f"Generating video for scenario: {scenario.scenario_id}")
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    images = []
    fig, ax = plt.subplots(figsize=(12, 12))

    num_timesteps = len(scenario.timestamps_seconds)
    # Only plot every 4th timestep
    frame_indices = list(range(0, num_timesteps, 1))
    for i in tqdm(frame_indices, desc="Rendering frames"):
        plot_timestep(scenario, i, ax, show_legend=False) # Use our plotting function on the same axes
        
        # Convert the plot to an image array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        
    plt.close(fig) # Close the figure to free memory

    # Create the animation
    print("Compiling video...")
    fig_anim, ax_anim = plt.subplots()
    ax_anim.axis('off') # Turn off axis for the final video
    
    def animate(i):
        ax_anim.imshow(images[i])
        return []
    
    anim = animation.FuncAnimation(
        fig_anim, animate, frames=len(images), interval=100, blit=True
    )
    
    # Save the video
    output_path = os.path.join(output_folder, f"{scenario.scenario_id}.mp4")
    # dpi controls the resolution of the saved video.
    
    print(f"Saving video to: {output_path}")
    progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')
    anim.save(output_path, writer='ffmpeg', dpi=150, progress_callback=progress_callback, fps=10)
    plt.close(fig_anim)
    
    print(f"Video saved successfully to: {output_path}")
    
# --- Main Inspection Logic ---

# First, a sanity check to make sure the file exists at the path.
if not os.path.exists(TFRECORD_FILE):
    raise FileNotFoundError(f"Could not find the file: {TFRECORD_FILE}")

print(f"--- Inspecting File: {TFRECORD_FILE} ---")
#%%
# 1. Create a TensorFlow Dataset object to read the file.
# This reads the file lazily, which is very memory efficient.
dataset = tf.data.TFRecordDataset(TFRECORD_FILE, compression_type='')

# 2. Iterate through the dataset. Each 'data' item is one serialized record (9-seconds scenario).
for i, data in enumerate(dataset):
    if i >= NUM_SCENARIOS_TO_INSPECT:
        break # Stop after inspecting the desired number of scenarios

    print(f"\n\n================ SCENARIO #{i+1} ================")
    
    # 3. Create an empty Scenario object from the protobuf definition.
    scenario = scenario_pb2.Scenario()
    
    # 4. Parse the serialized binary data into the 'scenario' object.
    # This is the "magic" decoding step.
    scenario.ParseFromString(data.numpy())

    # 5. Now, 'scenario' is a populated Python object we can inspect!
    # Let's print some high-level information.
    print(f"Scenario ID: {scenario.scenario_id}")
    print(f"Number of timestamps (steps): {len(scenario.timestamps_seconds)}")
    print(f"Number of agents (tracks) in scene: {len(scenario.tracks)}")
    print(f"Number of map features: {len(scenario.map_features)}")
    
    # Let's plot the first timestep (index 10) of this scenario.
    print(f"Plotting timestep {scenario.current_time_index} (current state)...")
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_timestep(scenario, scenario.current_time_index, ax, show_legend=True)
    plt.show()
    sdc_index = scenario.sdc_track_index

    print(f"Self-Driving Car (SDC) is at track index: {sdc_index}")

    # Let's look at the SDC track in more detail.
    if sdc_index >= 0 and sdc_index < len(scenario.tracks):
        sdc_track = scenario.tracks[sdc_index]
        print(f"\n--- SDC Track Details (ID: {sdc_track.id}) ---")
        
        # The data contains 10 past steps, 1 current step, and 80 future steps.
        # The "current" state is at index 10.
        current_state_index = 10
        current_state = sdc_track.states[current_state_index]
        
        print(f"CURRENT State (at T=0):")
        print(f"  > Position (x, y): ({current_state.center_x:.2f}, {current_state.center_y:.2f})")
        print(f"  > Velocity (vx, vy): ({current_state.velocity_x:.2f}, {current_state.velocity_y:.2f})")
        print(f"  > Heading (radians): {current_state.heading:.2f}")
        print(f"  > Dimensions (L, W, H): ({current_state.length:.2f}, {current_state.width:.2f}, {current_state.height:.2f})")
        print(f"  > State is valid: {bool(current_state.valid)}")

        # Let's look at a future state
        future_state_index = 30 # This is 2 seconds into the future (20 steps * 0.1s/step)
        future_state = sdc_track.states[future_state_index]
        print(f"FUTURE State (at T=+2s):")
        print(f"  > Position (x, y): ({future_state.center_x:.2f}, {future_state.center_y:.2f})")
        print(f"  > State is valid: {bool(future_state.valid)}")

print("\n--- Inspection Complete ---")

#%%
create_scenario_video(scenario, "videos/")
# %%
