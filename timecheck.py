import pybullet as p
import time

# Connect to the physics server
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version

# Set gravity
p.setGravity(0, 0, -9.8)

# Load a plane for the ground
#planeId = p.loadURDF("plane.urdf")

# Load the robot URDF file
robotStartPos = [0, 0, 0]  # Set the initial position of the robot
robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])  # Set the initial orientation of the robot
robotId = p.loadURDF("kuka_lbr_iiwa_support/urdf/lbr_iiwa_14_r820.urdf", robotStartPos, robotStartOrn)

# Set the joint positions
numJoints = p.getNumJoints(robotId)-2
initialPositions = [0, 0, 0, 0, 0, 0, 0]
targetPositions = [-1.191491470633566, 0.7985218401616503, 2.451986270233856, -2.6064809624268763, 2.640108453384974, -1.2483335772119015, 1.6007703609925172]

for jointIndex in range(numJoints):
    p.resetJointState(robotId, jointIndex, initialPositions[jointIndex])

# Set the desired target joint positions
#targetPositions = [0, 0, 0, 0, 0, 0, 0]  # Initialize target positions to zeros
duration = 5  # Duration in seconds
change=[]
for jointIndex in range(numJoints):
    #targetPositions[jointIndex] = jointPositions[jointIndex]  # Set initial target positions to starting configuration
    change.append((targetPositions[jointIndex] - initialPositions[jointIndex]) / duration)  # Calculate increment for each joint
print(change)

# Run the simulation for the desired duration
startTime = time.time()
while time.time() - startTime < duration:
    p.stepSimulation()
    for jointIndex in range(numJoints):
        p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, change)
    time.sleep(1. / 240.)  # Control the simulation speed

# Close the simulation
p.disconnect()
