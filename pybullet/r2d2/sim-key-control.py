import pybullet as p
import time
import pybullet_data

# Connect to the physics server (GUI mode)
p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Add search path for PyBullet URDFs

# real time
p.setRealTimeSimulation(1)

# Set gravity
p.setGravity(0, 0, -9.8)

# Load the plane and R2D2
planeId = p.loadURDF("plane.urdf")
r2d2Id = p.loadURDF("r2d2.urdf", [0, 0, 1])


# Export all joints info to csv

# df = pd.DataFrame(columns=['jointIndex', 'jointName', 'jointType', 'qIndex', 'uIndex', 'flags', 'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit', 'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis', 'parentFramePos', 'parentFrameOrn', 'parentIndex'])
# for joint in range(p.getNumJoints(r2d2Id)):
#     jointInfo = p.getJointInfo(r2d2Id, joint)
#     df = df._append({'jointIndex': jointInfo[0], 'jointName': jointInfo[1].decode('utf-8'), 'jointType': jointInfo[2], 'qIndex': jointInfo[3], 'uIndex': jointInfo[4], 'flags': jointInfo[5], 'jointDamping': jointInfo[6], 'jointFriction': jointInfo[7], 'jointLowerLimit': jointInfo[8], 'jointUpperLimit': jointInfo[9], 'jointMaxForce': jointInfo[10], 'jointMaxVelocity': jointInfo[11], 'linkName': jointInfo[12].decode('utf-8'), 'jointAxis': jointInfo[13], 'parentFramePos': jointInfo[14], 'parentFrameOrn': jointInfo[15], 'parentIndex': jointInfo[16]}, ignore_index=True)
# df.to_csv('pybullet/r2d2/r2d2_joints.csv', index=True)

# Get the number of joints of R2D2
numJoints = p.getNumJoints(r2d2Id)

targetVelocity = 0  # Target velocity
maxForce = 10  # Maximum force that can be applied to the joint
turn = 0  # Turn direction
increment = 0.5  # Increment for smoother movement

# Simulate indefinitely
while True:
    # Get key events
    keys = p.getKeyboardEvents()
    
    # Check if 'up' arrow key is pressed or held
    if p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN):
        targetVelocity -= increment  # Increase velocity
    
    # Check if 'down' arrow key is pressed or held
    if p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN):
        targetVelocity += increment  # Decrease velocity
    
    # Check if 'right' arrow key is pressed or held
    if p.B3G_RIGHT_ARROW in keys and (keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN):
        turn -= increment  # Turn right
    
    # Check if 'left' arrow key is pressed or held
    if p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN):
        turn += increment  # Turn left
    
    # Check if 'space' key is pressed and open the gripper
    if p.B3G_SPACE in keys and (keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED):
        for joint in range(numJoints):
            jointInfo = p.getJointInfo(r2d2Id, joint)
            if "_gripper" in jointInfo[1].decode('utf-8'):
                p.setJointMotorControl2(r2d2Id, joint, p.POSITION_CONTROL, targetPosition=0.5, force=maxForce)

    # Reset velocity and turn when keys are released
    if p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & p.KEY_WAS_RELEASED):
        targetVelocity = 0
    if p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & p.KEY_WAS_RELEASED):
        targetVelocity = 0
    if p.B3G_RIGHT_ARROW in keys and (keys[p.B3G_RIGHT_ARROW] & p.KEY_WAS_RELEASED):
        turn = 0
    if p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & p.KEY_WAS_RELEASED):
        turn = 0
        
    # Check if 'space' key is pressed and close the gripper
    if p.B3G_SPACE in keys and (keys[p.B3G_SPACE] & p.KEY_WAS_RELEASED):
        for joint in range(numJoints):
            jointInfo = p.getJointInfo(r2d2Id, joint)
            if "_gripper" in jointInfo[1].decode('utf-8'):
                p.setJointMotorControl2(r2d2Id, joint, p.POSITION_CONTROL, targetPosition=0, force=maxForce)
    # Ensure targetVelocity and turn are within limits
    targetVelocity = max(min(targetVelocity, 10), -10)
    turn = max(min(turn, 10), -10)

    # Iterate over all joints
    for joint in range(numJoints):
        # Get joint info
        jointInfo = p.getJointInfo(r2d2Id, joint)
        
        # Check if "_wheel_joint" is in the name of the joint
        if "_wheel_joint" in jointInfo[1].decode('utf-8'):
            # Control the joint
            if "left" in jointInfo[1].decode('utf-8'):
                p.setJointMotorControl2(r2d2Id, joint, p.VELOCITY_CONTROL, targetVelocity=targetVelocity + turn, force=maxForce)
            elif "right" in jointInfo[1].decode('utf-8'):
                p.setJointMotorControl2(r2d2Id, joint, p.VELOCITY_CONTROL, targetVelocity=targetVelocity - turn, force=maxForce)
    
    p.stepSimulation()
    time.sleep(1./240.)

# Disconnect from the server
p.disconnect()