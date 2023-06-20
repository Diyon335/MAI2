import random

import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt


robot_urdf = "kuka_lbr_iiwa_support/urdf/lbr_iiwa_14_r820.urdf"
simulation_steps = 1000


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

    time_step = 1/240
    p.setTimeStep(time_step)

    p_0 = None
    r_0 = None
    previous_velocities = None

    gain_values = [1.0, 0.5, 0.8, 1.0, 0.5, 0.8, 1.0]
    gain_matrix = np.diag(gain_values)

    residuals = []
    time_steps = []
    all_joint_velocities = []
    all_momenta = []

    # Display in the simulation how the robot goes from initial position, to our desired position
    for i in range(simulation_steps):

        p.stepSimulation()

        current_time = i * time_step
        time_steps.append(current_time)
        # print(f"Time step: {current_time}")

        # Retrieve joint positions and velocities
        joint_positions = []
        joint_velocities = []
        joint_names = []
        joint_types = []
        joint_indexes = []

        for joint_index in range(num_joints):
            joint_state = p.getJointState(robot, joint_index)
            joint_info = p.getJointInfo(robot, joint_index)

            j_type = joint_info[2]

            if j_type > 0:
                continue

            j_name = joint_info[1].decode("utf-8")
            j_index = joint_info[0]

            j_velocity = joint_state[1]
            j_position = joint_state[0]

            joint_positions.append(j_position)
            joint_velocities.append(j_velocity)
            joint_names.append(j_name)
            joint_indexes.append(j_index)
            joint_types.append(j_type)

        # print(joint_positions)
        # print(joint_velocities)
        # print(joint_names)
        # print(joint_types)
        # print(joint_indexes)

        ################################################################

        inertia_matrix = np.array(p.calculateMassMatrix(robot, joint_positions))

        # print(f"Inertia matrix: {inertia_matrix}")
        # print(inertia_matrix.shape)

        #################################################################

        p_t = np.dot(inertia_matrix, joint_velocities)

        if i == 0:

            p_0 = np.array(p_t)
            r_0 = np.array([0 for _ in range(len(joint_positions))])
            previous_velocities = [0 for _ in range(len(joint_positions))]

            residuals.append(r_0)

        # print(f"Initial Momentum: {p_0}")
        # print(f"Initial residual: {r_0}")
        # print(f"Current momentum: {p_t}")

        all_joint_velocities.append(joint_velocities)
        all_momenta.append(p_t)

        #################################################################

        joint_accelerations = [(v - prev_v) / time_step for v, prev_v in zip(joint_velocities, previous_velocities)]
        joint_accelerations = np.array(joint_accelerations)

        previous_velocities = joint_velocities

        # print(f"Joint accelerations: {joint_accelerations}")

        #################################################################
        # TODO: CHECK IF THIS WORKS
        C = compute_coriolis_matrix_v2(joint_positions, joint_velocities, joint_accelerations, robot)

        # print(f"Coriolis Matrix: {C}")
        # print(f"Coriolis matrix shape: {C.shape}")

        #################################################################

        gravity_matrix = calculateGravityMatrix(robot, joint_positions, joint_velocities, joint_accelerations)

        # print(f"Gravity matrix: {gravity_matrix}")
        # print(f"Gravity matrix shape: {gravity_matrix.shape}")

        if i == 100:
            # Retrieve the link positions and orientations
            link_states = p.getLinkStates(robot, range(7))

            link_positions = [state[0] for state in link_states]
            link_orientations = [state[1] for state in link_states]

            # Generate a random index to select a link
            random_index = random.randint(3, len(link_positions) - 1)

            # Get the position and orientation of the randomly selected link
            selected_link_position = link_positions[random_index]
            selected_link_orientation = link_orientations[random_index]

            force = [400, 550, 700]  # Define the force vector (e.g., 10 N in the x-direction)
            position = [0, 0, 0]  # Define the position of the external force (e.g., at the joint's origin)
            p.applyExternalForce(robot, random_index, force, position, p.LINK_FRAME)  # Apply the force

            print("FORCE APPLIED AT LINK", random_index)
            print(f"FORCE APPLIED AT TIME: {current_time}")
            time.sleep(5)

            # Create a visual marker at the selected point
            marker_size = 0.05
            marker_color = [1, 0, 0, 1]  # Red color
            marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=marker_size, rgbaColor=marker_color)
            marker_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=marker_size)

            marker_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_visual,
                                            baseCollisionShapeIndex=marker_collision,
                                            basePosition=selected_link_position,
                                            baseOrientation=selected_link_orientation)

        ##################### Algorithm ################################

        tau = np.dot(inertia_matrix, joint_accelerations) + np.dot(C, joint_velocities) + gravity_matrix

        # print(f"Torque: {tau}")

        last_residual = residuals[-1]

        integral_sum = tau + np.dot(np.transpose(C), joint_velocities) - gravity_matrix + np.array(last_residual)
        integral = integral_sum * current_time

        final_sum = p_t - integral - p_0

        new_residual = np.dot(gain_matrix, final_sum)

        residuals.append(new_residual)

        print(f"New residual: {new_residual}")

        #####################################################

        # TODO: Stop the simulation based on residuals

        if current_time > 1.0:
            break

    # Remove initial residual value
    residuals.pop(0)

    plot_graph("Plot of residuals over time",
               "residuals",
               "Residual",
               residuals,
               time_steps)

    plt.clf()

    plot_graph("Plot of momenta over time",
               "momenta",
               "Momentum (kg/ms-2)",
               all_momenta,
               time_steps)

    plt.clf()

    plot_graph("Plot of velocity over time",
               "velocities",
               "Velocity (ms-1)",
               all_joint_velocities,
               time_steps)

    p.disconnect()


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


def compute_coriolis_matrix_v2(joint_positions, joint_velocities, joint_accelerations, robot):
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


def calculateGravityMatrix(robot, joint_positions, joint_velocities, joint_accelerations):
    num_joints = p.getNumJoints(robot)
    # joint_velocities = [0.0] * num_joints  # Zero joint velocities
    # joint_accelerations = [0.0] * num_joints  # Zero joint accelerations

    joint_forces = calculateInvDynamics(joint_positions, joint_velocities, joint_accelerations, robot)

    gravity_matrix = np.array(joint_forces)

    return gravity_matrix


def plot_graph(title, figure_name, variable_name, variables, time_steps):

    vel = np.array(variables)
    timesteps = np.array(time_steps)

    for joint in range(7):
        plt.plot(timesteps, vel[:, joint], label=f'Joint {joint + 1}')

    plt.xlabel('Time (s)')
    plt.ylabel(variable_name)
    plt.title(title)
    plt.legend()

    plt.savefig(figure_name+".png")


if __name__ == '__main__':
    main()
