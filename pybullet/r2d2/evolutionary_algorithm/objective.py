import pybullet as p
import pybullet_data
import numpy as np
import json
import os
import datetime

# R2D2 is driven as a differential-drive base: its four wheels are grouped into
# a left and a right side. Both wheels on a side always share one velocity, so
# the only motors worth controlling are "left side" and "right side". The URDF
# exposes the wheels at these joint indices (see specs/r2d2_joints.csv).
LEFT_WHEEL_JOINT_INDICES = (6, 7)   # left_front_wheel_joint, left_back_wheel_joint
RIGHT_WHEEL_JOINT_INDICES = (2, 3)  # right_front_wheel_joint, right_back_wheel_joint

# The genome is just two values: [left_side_velocity, right_side_velocity].
# Equal values drive straight; a difference steers. Nothing else on the robot
# (legs, gripper, head) is actuated, so the search space is only these two DOF.
GENOME_SIZE = 2

# Target the robot should try to reach (placed directly forward of the start).
TARGET_POSITION = [0, 10, 0]


def apply_wheel_velocities(robot_id, robot_parameters, force=500):
    """Drive R2D2 as a differential-drive base from a two-gene genome.

    The genome holds one target velocity per side, ``[left, right]``. Both
    wheels on a side are driven at that side's velocity, so equal values go
    straight and a difference turns. Shared by the training objective and the
    best-solution replay so both apply the genome the same way.

    Args:
        robot_id: PyBullet body id of the loaded robot.
        robot_parameters: two-element vector ``[left_velocity, right_velocity]``.
        force: maximum motor force applied to each wheel joint.
    """
    left_velocity, right_velocity = robot_parameters[0], robot_parameters[1]
    for joint_index in LEFT_WHEEL_JOINT_INDICES:
        p.setJointMotorControl2(robot_id, joint_index, p.VELOCITY_CONTROL,
                                targetVelocity=left_velocity, force=force)
    for joint_index in RIGHT_WHEEL_JOINT_INDICES:
        p.setJointMotorControl2(robot_id, joint_index, p.VELOCITY_CONTROL,
                                targetVelocity=right_velocity, force=force)


def save_training_data(data, filename):
    """
    Save the training data to a file.

    Args:
        data (dict): The training data to be saved.
        filename (str): The name of the file to save the data to.

    Returns:
        None
    """
    directory = os.path.dirname(filename)
    
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Add the current timestamp to the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename_with_timestamp = f"{filename}_{timestamp}"
    
    # Now save the data
    with open(filename_with_timestamp, "w") as f:
        json.dump(data, f)

def objective_function(robot_parameters):
    """
    Evaluates the performance of a robot with given parameters in a PyBullet simulation.

    Args:
        robot_parameters (list): A list of parameters that define the robot's behavior.

    Returns:
        float: The negative distance between the robot and a target position in the simulation.
    """
    # Connect to the physics server. Training runs headless (DIRECT) — the
    # OpenGL GUI is far slower and is only used for the final best-solution
    # replay. Reuse any connection the caller (or worker process) already has.
    if p.getConnectionInfo()['isConnected'] == 0:
        p.connect(p.DIRECT)

    # Reset the simulation
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # To load URDF from PyBullet's data path
    p.setGravity(0, 0, -9.8)
    
    # Load the plane
    p.loadURDF("plane.urdf")

    # Set the robot's initial position above the plane
    start_position = [0, 0, 1]  # Adjust z value as needed
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])

    # Load the robot
    robot_id = p.loadURDF("r2d2.urdf", start_position, start_orientation)

    apply_wheel_velocities(robot_id, robot_parameters)

    # A failed simulation (e.g. the physics server dropped) should score as the
    # worst possible candidate rather than crash the run with a NameError, so
    # seed the distance with a large penalty before the simulation block.
    distance_to_target = 1e6

    # Run the simulation for a fixed number of steps.
    try:
        steps = 6500

        # Target the robot should try to reach (directly forward).
        target_position = TARGET_POSITION

        # Add a small floating red circle at the target position
        target_visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.25, rgbaColor=[0, 1, 0, 1])
        target_collision_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=target_collision_shape_id, baseVisualShapeIndex=target_visual_shape_id, basePosition=target_position)
        
        
        for _ in range(steps):
            p.stepSimulation()

        # Code to evaluate the robot's performance goes here...
        
        
        robot_position = p.getBasePositionAndOrientation(robot_id)[0]
        distance_to_target = np.linalg.norm(np.array(target_position) - np.array(robot_position))

    except p.error as e:
        if "Not connected to physics server" in str(e):
            # Save the most recent training data before handling the error
            save_training_data(robot_parameters, "r2d2/evolutionary_algorithm/trained_models/latest_training_data.json")
            print("Error: Not connected to physics server. Latest training data saved.")
            # Handle the error, e.g., attempt to reconnect, log the error, or skip this simulation
        else:
            raise  # Re-raise the exception if it's not the specific error we're handling
   
    return -distance_to_target  # The negative distance (we want to minimize distance)