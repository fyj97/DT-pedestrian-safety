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
        
        # Prediction (warning) is `warning`
        prediction = ped.get('warning', False)
        
        if ground_truth and prediction:
            TP += 1  # True Positive
        elif not ground_truth and not prediction:
            TN += 1  # True Negative
        elif not ground_truth and prediction:
            FP += 1  # False Positive
        elif ground_truth and not prediction:
            FN += 1  # False Negative
    
    # Create confusion matrix as a 2x2 numpy array
    confusion_matrix = np.array([[TP, FP], [FN, TN]])
    
    return confusion_matrix

def calculate_tpr_fpr(confusion_matrix):
    """Calculate True Positive Rate (Sensitivity) and False Positive Rate (1-Specificity)"""
    TP, FP = confusion_matrix[0]
    FN, TN = confusion_matrix[1]
    
    # Avoid division by zero
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    return TPR, FPR

def run_simulation_with_threshold(filename, danger_threshold):
    """Run the simulation with a specific danger threshold and return all spawned pedestrians"""
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
            update_simulation(message, active_pedestrians, danger_threshold)

        # Move pedestrians along their trajectories
        for ped in active_pedestrians:
            if iteration % 5 == 0 and ped['step'] < len(ped['traj']):  # Move every 5 iterations to slow movement
                ped['step'] += 1  # Move to the next step

        # Remove pedestrians that have reached the end of their trajectory
        active_pedestrians = [ped for ped in active_pedestrians if ped['step'] < len(ped['traj'])]

        iteration += 1  # Increment iteration count
        time.sleep(0.01)  # Faster simulation for multiple runs
    
    return all_spawned_pedestrians

def update_simulation(message, pedestrians, danger_threshold):
    """Update simulation without visualization for faster processing"""
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
    
    # Check future collision prediction - VARIABLE threshold for ROC analysis
    check_future_collision(pedestrians, vehicles, danger_threshold)

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
# Function to check if a pedestrian might collide with any vehicle in the future for the entire trajectory
def check_future_collision(pedestrians, vehicles, danger_threshold=200):
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
            check_steps_ahead = 12
            max_steps = min(remaining_ped_steps, remaining_vehicle_steps, check_steps_ahead)

            # Compare future points along the trajectory
            for look_ahead in range(1, max_steps):
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
                
                # If the distance is below the threshold, mark the pedestrian as having a warning
                if distance < danger_threshold:
                    ped['warning'] = True
                    break  # No need to check further if a warning is already set
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
    # Test thresholds from 10 to 200 in smaller steps for smoother ROC curve
    thresholds = list(range(10, 201, 5))  # Step by 5 instead of 10 for more granularity
    print(f"Testing {len(thresholds)} thresholds: {thresholds}")
    
    # Store results
    confusion_matrices = []
    tprs = []
    fprs = []
    
    # Run simulation for each threshold
    for i, threshold in enumerate(thresholds):
        print(f"Processing threshold {threshold} ({i+1}/{len(thresholds)})")
        
        # Reset random seed for consistent results across thresholds
        random.seed(42)
        
        # Run simulation with current threshold
        all_pedestrians = run_simulation_with_threshold(filename, threshold)
        
        # Calculate confusion matrix
        cm = calculate_confusion_matrix(all_pedestrians)
        confusion_matrices.append(cm)
        
        # Calculate TPR and FPR
        tpr, fpr = calculate_tpr_fpr(cm)
        tprs.append(tpr)
        fprs.append(fpr)
        
        print(f"  Threshold {threshold}: TPR={tpr:.3f}, FPR={fpr:.3f}")
        print(f"  Confusion Matrix: TP={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TN={cm[1,1]}")
    
    # Save confusion matrices to file
    np.save('dd_confusion_matrices.npy', np.array(confusion_matrices))
    print(f"Saved {len(confusion_matrices)} confusion matrices to 'dd_confusion_matrices.npy'")

# Call main function with your filename
filename = 'mqtt_messages.txt'  # Replace with your actual file path
main(filename)
