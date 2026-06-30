import pybullet as p
import pybullet_data

def visualize_best_solution(best_parameters):
    """
    Visualizes the best solution by connecting to the physics server, resetting the simulation,
    loading the plane and robot, applying the best parameters to the robot joints, running the
    simulation for a certain number of steps, and then disconnecting from the physics server.

    Args:
        best_parameters (list): A list of parameters representing the desired positions of the robot joints.

    Returns:
        None
    """
    # Connect to the physics server
    if p.getConnectionInfo()['isConnected'] == 0:
        p.connect(p.GUI)

    # Reset the simulation
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # Load the plane
    p.loadURDF("plane.urdf")

    # Set the robot's initial position above the plane
    start_position = [0, 0, 1]  # Adjust z value as needed
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
    # Load the robot
    robot_id = p.loadURDF("r2d2.urdf", start_position, start_orientation)
    
    # Apply the parameters to the robot joints
    for joint_index, param in enumerate(best_parameters):
        p.setJointMotorControl2(robot_id, joint_index, p.POSITION_CONTROL, targetPosition=param)
    
    # Run the simulation for a certain number of steps
    for _ in range(1000):
        p.stepSimulation()

    # Disconnect from the physics server
    p.disconnect()

# Example usage:
# best_parameters = np.array([...])
# visualize_best_solution(best_parameters)