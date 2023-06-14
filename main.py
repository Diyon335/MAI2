import random

import pybullet as p
import time
import pybullet_data
import numpy as np
import sympy as sp
import faulthandler
faulthandler.enable()
from scipy.integrate import quad


robot_urdf = "kuka_lbr_iiwa_support/urdf/lbr_iiwa_14_r820.urdf"





# Define the function to integrate
def integrand(a,b,q, q_dot, torque_tot, C, r):
    # Define the expression for the integrand based on your problem
    result = torque_tot + C @ q_dot + r
    return result




def calculateInvDynamics(joint_positions, joint_velocities, joint_accelerations, robot_id):
    num_joints = len(joint_positions)
    joint_forces = np.zeros(num_joints)

    for i in range(num_joints):
        joint_position = joint_positions[i]
        joint_velocity = joint_velocities[i]
        joint_acceleration = joint_accelerations[i]

        joint_info = p.getJointInfo(robot_id, i)
        joint_mass = joint_info[10]  # Index 10 corresponds to the joint mass
        joint_inertia = joint_info[11]  # Index 11 corresponds to the joint inertia

        # Compute Coriolis and centrifugal forces (set to zero in this example)
        coriolis_centrifugal = 0.0

        # Compute gravitational forces
        gravitational_force = joint_mass * 9.8 * np.sin(joint_position)

        # Compute joint forces using Newton-Euler dynamics equations
        joint_forces[i] = joint_inertia * joint_acceleration + coriolis_centrifugal + gravitational_force

    return joint_forces



def compute_coriolis_matrix_v2(joint_positions, joint_velocities, joint_accelerations,robot):
    # Demux the input variable
    # q = qdq[:len(qdq)//2]
    # dq = qdq[len(qdq)//2:]

    # Calculation of the Coriolis matrix using the method by Corke
    N = len(joint_positions)
    C = np.zeros((N, N))
    Csq = np.zeros((N, N))

    for j in range(N):
        QD = np.zeros(N)
        QD[j] = 1
        tau = calculateInvDynamics(joint_positions, joint_velocities, joint_accelerations, robot)
        Csq[:, j] = Csq[:, j] + tau

    for j in range(N):
        for k in range(j + 1, N):
            QD = np.zeros(N)
            QD[j] = 1
            QD[k] = 1
            tau = calculateInvDynamics(joint_positions, joint_velocities, joint_accelerations, robot)
            C[:, k] = C[:, k] + (tau - Csq[:, k] - Csq[:, j]) * joint_velocities[j]

    C = C + np.dot(Csq, np.diag(joint_velocities))

    return C

    

def calculateGravityMatrix(robot, joint_positions,joint_velocities,joint_accelerations):
    num_joints = p.getNumJoints(robot)
    # joint_velocities = [0.0] * num_joints  # Zero joint velocities
    # joint_accelerations = [0.0] * num_joints  # Zero joint accelerations

    joint_forces = calculateInvDynamics(joint_positions, joint_velocities, joint_accelerations,robot)
    
    gravity_matrix = np.array(joint_forces)
    
    return gravity_matrix


def main():

    # Connect to the physics server
    physicsClient = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set gravity
    p.setGravity(0, 0, -9.81)

    # Load the plane
    plane = p.loadURDF("plane.urdf")

    # Load the robot
    base_position = [0, 0, 0]
    robot = p.loadURDF(robot_urdf, base_position, useFixedBase=1)

    # Get the number of joints
    num_joints = p.getNumJoints(robot)
    print(num_joints)

    # Give the final orientation and joint positions of the robot
    final_orientation = p.getQuaternionFromEuler([3.14, 0, 0])
    final_joint_positions = p.calculateInverseKinematics(robot,
                                                         7,
                                                         [0.1, 0.1, 0.4],
                                                         targetOrientation=final_orientation
                                                         )

    # Set all joints to the final desired position
    p.setJointMotorControlArray(robot,
                                range(7),
                                p.POSITION_CONTROL,
                                targetPositions=final_joint_positions
                                )

    prev_joint_velocities = [0.0] * num_joints

    # Display in the simulation how the robot goes from initial position, to our desired position
    for i in range(100):
        #print(i)
        p.stepSimulation()

        # Retrieve joint positions and velocities
        joint_positions = []
        joint_velocities = []
        joint_names = []
        joint_dof = []
        #print(num_joints)
        for joint_index in range(num_joints):
            joint_state = p.getJointState(robot, joint_index)
            joint_info = p.getJointInfo(robot, joint_index)
            joint_names.append(joint_info[1].decode("utf-8"))  # Decode joint name from bytes to string
            joint_dof.append(joint_info[3])
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])

        
        #print(f"Joint names: {joint_names}")
        #print(f"Joint dofs: {joint_dof}")
       
        fixed_joints = joint_dof.count(-1)
        #joint_velocities = joint_velocities[:-fixed_joints]
        #joint_positions = joint_positions[:-fixed_joints]

        #print(f"Joint positions: {joint_positions}")
        #print(f"Joint velocities: {joint_velocities}")

        ###############################################################
        # Retrieve the inertia matrix for each joint
        inertia_matrices = []
        for joint_index in range(num_joints):
            dynamics_info = p.getDynamicsInfo(robot, joint_index)
            inertia_matrix = list(dynamics_info[2])

            while len(inertia_matrix) < num_joints:
                inertia_matrix.append(0)

            inertia_matrix = np.array(inertia_matrix)

            inertia_matrices.append(inertia_matrix)

        inertia_matrices = np.array(inertia_matrices)
        #print("Inertia: ",np.array(inertia_matrices).shape)
        # joint_velocities = np.transpose(joint_velocities)

        ################################################################

        # Compute p(t) as M * q_dot
        #print("Calculating momentum")
        p_t = np.dot(inertia_matrices, np.transpose(joint_velocities))
        #print(f"p(t): {p_t}")
        if i == 0: 
            p_0 = p_t
        ################################################################
        # Compute joint accelerations using finite difference method
        dt = 1.0 / 240.0  # Time step
        joint_accelerations = [(v - prev_v) / dt for v, prev_v in zip(joint_velocities, prev_joint_velocities)]

        joint_accelerations = np.transpose(joint_accelerations)

        #print(f"Joint accelerations: {joint_accelerations}")

        prev_joint_velocities = joint_velocities

        #print("compute_coriolis_matrix")
        C = compute_coriolis_matrix_v2(joint_positions, joint_velocities, joint_accelerations,robot)
        #print(C)
        #print(C.shape)

        ################################################################

        # ################################################################
        # # Compute the gravity matrix
        # zero_velocities = np.zeros(num_joints)
        #print("Computing gravity matrix")
        gravity_matrix = calculateGravityMatrix(robot, joint_positions,joint_velocities, joint_accelerations)
        
        #print(f"Gravity matrix: {gravity_matrix}")
       # print(gravity_matrix.shape)

        #################### Algorithm ################################
        
        #Computing torque
        tau = np.dot(inertia_matrices,joint_accelerations) + np.dot(C,joint_velocities) + gravity_matrix
        #print('Torque:', tau)

        # Create the diagonal gain matrix using np.diag
        gain_values = [1.0, 0.5, 0.8,1.0, 0.5, 0.8,1.0, 0.5, 0.8]  # Example gain values for three joints
        gain_matrix = np.diag(gain_values)

        # Specify the integration limits
        a = 0
        b = i
        r = p_t - p_0
        #print ("RESULT = ",r)
        # Call the quad function to perform integration
        #result, error = quad(integrand, a, b, args=(b, joint_positions, joint_velocities, tau, C, result))

        # Print the result
        #print("Integration result:", result)
        #print("Estimated error:", error)

        
        # This is just a placeholder to generate a collision at a random link of the robot
        if i == 4:
            # Retrieve the link positions and orientations
            link_states = p.getLinkStates(robot, range(p.getNumJoints(robot)))

            link_positions = [state[0] for state in link_states]
            link_orientations = [state[1] for state in link_states]

            print(len(link_positions))
            # Generate a random index to select a link
            random_index = random.randint(3, len(link_positions)-1)
            #print(random_index)
            # Get the position and orientation of the randomly selected link
            selected_link_position = link_positions[random_index]
            selected_link_orientation = link_orientations[random_index]


            #random_joint_index = p.getRandomJointIndices(robot, 1)[0]  # Get a random joint index
            force = [400, 550, 700]  # Define the force vector (e.g., 10 N in the x-direction)
            position = [0, 0, 0]  # Define the position of the external force (e.g., at the joint's origin)
            p.applyExternalForce(robot, random_index, force, position, p.LINK_FRAME)  # Apply the force
            print("FORCE APPLIED AT LINK" , random_index)

            # Create a visual marker at the selected point
            marker_size = 0.05
            marker_color = [1, 0, 0, 1]  # Red color
            marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=marker_size, rgbaColor=marker_color)
            marker_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=marker_size)

            marker_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_visual,
                                            baseCollisionShapeIndex=marker_collision,
                                            basePosition=selected_link_position,
                                            baseOrientation=selected_link_orientation)

        time.sleep(1. / 10.)

    p.disconnect()


if __name__ == '__main__':
    main()
