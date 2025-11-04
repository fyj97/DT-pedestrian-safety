import json
import random
import time
import math
import numpy as np


# Predefined fixed points A, B, C, D
fixed_points = {
    'A': (120, 650),
    'B': (660, 650),
    'C': (120, 100),
    'D': (660, 100)
}

def calculate_confusion_matrix(pedestrians):
    TP = TN = FP = FN = 0
    
    for ped in pedestrians:
        # Ground truth (collision) is `in_danger`
        ground_truth = ped.get('in_danger', False)
        
        # Prediction (warning) is `warning` - issued when collision is close enough
        # Distant predictions are not counted as positive predictions
        prediction = ped.get('warning', False) and not ped.get('distant_prediction', False)
        
        if ground_truth and prediction:
            TP += 1  # True Positive - collision occurred and was predicted when close enough
        elif not ground_truth and not prediction:
            TN += 1  # True Negative - no collision and no warning
        elif not ground_truth and prediction:
            FP += 1  # False Positive - warning given but no actual collision
        elif ground_truth and not prediction:
            FN += 1  # False Negative - collision occurred but no warning given when close enough
            # Note: distant predictions are counted as false negatives, not false positives
    
    # Create confusion matrix as a 2x2 numpy array
    confusion_matrix = np.array([[TP, FP], [FN, TN]])
    
    return confusion_matrix

def run_simulation_with_ttc(filename, ttc_look_ahead, ttc_threshold=6):
    """Run the simulation with a specific TTC look_ahead value and return all spawned pedestrians"""
    # List to hold all spawned pedestrians
    all_spawned_pedestrians = []
    
    # List to hold currently active pedestrians
    active_pedestrians = []
    
    pedestrian_spawn_time = random.uniform(1, 5)  # Initial spawn delay
    iteration = 0  # To control the movement speed

    for line in read_file_continuously(filename):
        message = parse_message(line)

        # Check if it's time to generate a new pedestrian
        pedestrian_spawn_time -= 0.1  # Reduce time by small increments
        if pedestrian_spawn_time <= 0:
            new_pedestrian = generate_virtual_pedestrian()
            active_pedestrians.append(new_pedestrian)
            all_spawned_pedestrians.append(new_pedestrian)  # Store in the list of all pedestrians
            pedestrian_spawn_time = random.uniform(1, 5)  # Set next spawn delay

        # Update the simulation with the message and active pedestrians
        if message:
            update_simulation_ttc(message, active_pedestrians, ttc_look_ahead, ttc_threshold)

        # Move pedestrians along their trajectories
        for ped in active_pedestrians:
            if iteration % 5 == 0 and ped['step'] < len(ped['traj']):  # Move every 5 iterations to slow movement
                ped['step'] += 1  # Move to the next step

        # Remove pedestrians that have reached the end of their trajectory
        active_pedestrians = [ped for ped in active_pedestrians if ped['step'] < len(ped['traj'])]

        iteration += 1  # Increment iteration count
        time.sleep(0.01)  # Faster simulation for multiple runs
    
    return all_spawned_pedestrians

def update_simulation_ttc(message, pedestrians, ttc_look_ahead, ttc_threshold=6):
    """Update simulation without visualization for faster processing - TTC version"""
    # Extracting bounding boxes, labels, trajectories, and IDs
    boxes = message['boxes']
    labels = message['labels']
    traj_tjs = message['traj_tjs']
    traj_ids = message['traj_id']  # IDs associated with each trajectory
    ids = message['ids']  # IDs associated with each label/box

    # Prepare list of vehicles with bounding box info and step info
    vehicles = []
    for i, label in enumerate(labels):
        if "vehicle" in label:  # Only consider vehicles
            vehicle_step = 0  # You may want to store the actual vehicle step if available
            vehicles.append({
                "box": boxes[i],
                "id": ids[i],
                "traj": traj_tjs[traj_ids.index(ids[i])] if ids[i] in traj_ids else [],
                "step": vehicle_step  # Add step for future trajectory prediction
            })

    # Check pedestrian proximity to vehicles (ground truth) - FIXED threshold at 50
    check_pedestrian_proximity(pedestrians, vehicles, danger_threshold=50)
    
    # Check future collision prediction - FIXED threshold at 50, VARIABLE TTC look_ahead
    check_future_collision_ttc(pedestrians, vehicles, danger_threshold=50, ttc_look_ahead=ttc_look_ahead, ttc_threshold=ttc_threshold)

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to check if a pedestrian is close to any vehicle and mark them as "in danger"
def check_pedestrian_proximity(pedestrians, vehicles, danger_threshold=50):
    for ped in pedestrians:
        # Skip pedestrians already marked as "in danger"
        if ped.get('in_danger', False):
            continue
        
        if ped['step'] < len(ped['traj']):  # Ensure pedestrian is still on the path
            ped_pos = ped['traj'][ped['step']]
            for vehicle in vehicles:
                # Calculate the center of the vehicle's bounding box
                box = vehicle['box']
                vehicle_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                
                # Calculate the distance between the pedestrian and the vehicle
                distance = calculate_distance(ped_pos, vehicle_center)
                
                # Check if the distance is less than the danger threshold
                if distance < danger_threshold:
                    ped['in_danger'] = True
                    break  # Mark the pedestrian in danger and stop checking other vehicles


# Function to continuously read the file and yield new lines
def read_file_continuously(filename, timeout=10):
    """Read the file continuously, yield new lines, and stop if no new data is received for a specified timeout."""
    with open(filename, "r") as file:
        last_activity_time = time.time()
        while True:
            line = file.readline()
            if not line:
                time.sleep(0.1)  # Wait before checking for new lines
                # Check if we have reached the timeout without new data
                if time.time() - last_activity_time > timeout:
                    print("No new data received for a while, stopping.")
                    break  # Exit the loop after the timeout
                continue
            last_activity_time = time.time()  # Reset the timeout on new data
            yield line


# Function to parse and extract relevant fields
def parse_message(line):
    try:
        message = line.split('Message: ')[1].strip()
        return json.loads(message)
    except (IndexError, json.JSONDecodeError):
        return None


# Function to check if a pedestrian might collide with any vehicle in the future for the entire trajectory
def check_future_collision_ttc(pedestrians, vehicles, danger_threshold=50, ttc_look_ahead=12, ttc_threshold=6):
    """
    Check for future collisions with look-ahead steps.
    Issue warning if Time to Collision (TTC) is within ttc_threshold steps (collision is close enough).
    """
    for ped in pedestrians:
        # Skip if pedestrian is already in a warning state
        if ped.get('warning', False):
            continue
        
        # Get the remaining trajectory length for the pedestrian
        remaining_ped_steps = len(ped['traj']) - ped['step']
        
        # For each vehicle, check the future positions
        for vehicle in vehicles:
            # Get the remaining trajectory length for the vehicle
            remaining_vehicle_steps = len(vehicle['traj']) - vehicle['step']
            
            # Determine the maximum steps to check (whichever trajectory is shorter)
            max_steps = min(remaining_ped_steps, remaining_vehicle_steps, ttc_look_ahead)

            # Compare future points along the trajectory
            for look_ahead in range(1, max_steps + 1):  # Include the ttc_look_ahead value
                # Get future pedestrian position
                if ped['step'] + look_ahead < len(ped['traj']):
                    ped_future_pos = ped['traj'][ped['step'] + look_ahead]
                else:
                    continue  # Skip if out of range

                # Get future vehicle position
                if vehicle['step'] + look_ahead < len(vehicle['traj']):
                    vehicle_future_pos = vehicle['traj'][vehicle['step'] + look_ahead]
                else:
                    continue  # Skip if out of range
                
                # Calculate the distance between the predicted future positions of the pedestrian and vehicle
                distance = calculate_distance(ped_future_pos, vehicle_future_pos)
                
                # If the distance is below the threshold, check if TTC is close enough to warn
                if distance < danger_threshold:
                    # TTC = look_ahead (steps until collision)
                    ttc = look_ahead
                    
                    # Issue warning if TTC is within threshold (collision is close enough)
                    if ttc <= ttc_threshold:
                        ped['warning'] = True
                        ped['ttc'] = ttc  # Store the Time to Collision
                        break  # No need to check further if a warning is already set
                    else:
                        # If TTC is too large, mark as distant prediction
                        ped['distant_prediction'] = True
                        ped['distant_ttc'] = ttc
            if ped['warning']:
                break  # Stop checking other vehicles if warning is set


# Function to generate random virtual pedestrians
def generate_virtual_pedestrian():
    # Possible paths
    paths = [
        ('A', 'B'), ('B', 'A'),
        ('C', 'A'), ('A', 'C'),
        ('C', 'D'), ('D', 'C'),
        ('D', 'B'), ('B', 'D')
    ]
    
    # Randomly select a start and end point from allowed paths
    start, end = random.choice(paths)
    start_pos = fixed_points[start]
    end_pos = fixed_points[end]
    
    # Generate a simple straight-line trajectory from start to end
    traj = [(
        start_pos[0] + (end_pos[0] - start_pos[0]) * i / 30,  # Slower movement with 30 points
        start_pos[1] + (end_pos[1] - start_pos[1]) * i / 30
    ) for i in range(31)]  # 31 points in the trajectory
    
    return {
        "start": start,
        "end": end,
        "traj": traj,
        "step": 0,  # Track how many steps have been made for this pedestrian
        "id": f"pedestrian_{random.randint(100, 999)}",
        "in_danger": False,  # Initialize in_danger flag to False
        "warning": False  # Initialize warning flag to False
    }


def main(filename):
    # Test TTC threshold values from 1 to 12 (since TTC look_ahead is 12)
    threshold_values = list(range(1, 13))
    print(f"Testing {len(threshold_values)} TTC threshold values: {threshold_values}")
    print(f"TTC look_ahead fixed at: 12")
    
    # Store results
    confusion_matrices = []
    
    # Run simulation for each threshold value
    for i, threshold in enumerate(threshold_values):
        print(f"Processing TTC threshold {threshold} ({i+1}/{len(threshold_values)})")
        
        # Reset random seed for consistent results across threshold values
        random.seed(42)
        
        # Run simulation with current threshold value (TTC look_ahead fixed at 12)
        all_pedestrians = run_simulation_with_ttc(filename, ttc_look_ahead=12, ttc_threshold=threshold)
        
        # Calculate confusion matrix
        cm = calculate_confusion_matrix(all_pedestrians)
        confusion_matrices.append(cm)
        
        print(f"  Completed TTC threshold {threshold}")
    
    # Save confusion matrices to file
    np.save('ttc_threshold_confusion_matrices.npy', np.array(confusion_matrices))
    print(f"\nSaved {len(confusion_matrices)} confusion matrices to 'ttc_threshold_confusion_matrices.npy'")

# Call main function with your filename
filename = 'mqtt_messages.txt'  # Replace with your actual file path
main(filename)  # Test all TTC threshold values from 1 to 12
