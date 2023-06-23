import copy
import random

import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


robot_urdf = "kuka_lbr_iiwa_support/urdf/lbr_iiwa_14_r820.urdf"
collision_index = 100


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

    simulation_steps = 1000

    p_0 = None
    r_0 = None
    previous_velocities = None
    selected_link = None

    gain_values = [0.01] * 7
    gain_matrix = np.diag(gain_values)
    # print(gain_matrix)

    residuals = []
    time_steps = []
    all_joint_velocities = []
    all_momenta = []
    all_link_positions = {i: [] for i in range(num_joints)}
    kinetic_energy = []

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
        joint_velocities_3D = []

        for joint_index in range(num_joints):
            link_state = p.getLinkState(robot, joint_index, computeLinkVelocity=1, computeForwardKinematics=1)
            joint_state = p.getJointState(robot, joint_index)
            joint_info = p.getJointInfo(robot, joint_index)

            j_type = joint_info[2]

            if j_type > 0:
                continue

            link_pos = link_state[4]
            all_link_positions[joint_index].append(link_pos)

            j_name = joint_info[1].decode("utf-8")
            j_index = joint_info[0]

            j_velocity = joint_state[1]
            j_position = joint_state[0]
            j_velocity_3D = link_state[6]

            joint_positions.append(j_position)
            joint_velocities.append(j_velocity)
            joint_names.append(j_name)
            joint_indexes.append(j_index)
            joint_types.append(j_type)
            joint_velocities_3D.append(j_velocity_3D)

        # print(joint_positions)
        # print(joint_velocities)
        # print(joint_names)
        # print(joint_types)
        # print(joint_indexes)
        # print(joint_velocities_3D)

        ################################################################

        inertia_matrix = np.array(p.calculateMassMatrix(robot, joint_positions))

        # print(f"Inertia matrix: {inertia_matrix}")
        # print(inertia_matrix.shape)

        #################################################################

        ke = 0.5 * np.matrix(joint_velocities) * inertia_matrix * np.matrix(joint_velocities).transpose()
        kinetic_energy.append(ke.item(0, 0))

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
        C = compute_coriolis_matrix(robot, joint_positions, joint_velocities, joint_accelerations)

        # print(f"Coriolis Matrix: {C}")
        # print(f"Coriolis matrix shape: {C.shape}")

        ########################## Coriolis Matrix Check #################

        # matches = []
        # affected_joints = [0, 3, 5]
        # delta_q = 0.001
        # dq = copy.copy(joint_positions)
        #
        # for joint in affected_joints:
        #     dq[joint] += delta_q
        #
        # dq_ = [q + q_ for q, q_ in zip(joint_positions, dq)]
        # M_k = inertia_matrix / dq_
        #
        # dM_dt = np.zeros(shape=M_k.shape)
        #
        # rows, cols = M_k.shape
        #
        # coriolis_check = C + np.transpose(C)
        #
        # for row in range(rows):
        #     for col in range(cols):
        #         dM_dt[row][col] = (inertia_matrix[row][col] - M_k[row][col]) / time_step
        #
        #         if dM_dt[row][col] == coriolis_check[row][col]:
        #             matches.append(True)
        #         else:
        #             matches.append(False)
        #
        # print(f"Coriolis computation is valid: {all(matches)}")

        #################################################################

        if i == collision_index:
            # Retrieve the link positions and orientations
            link_states = p.getLinkStates(robot, range(7))

            link_positions = [state[0] for state in link_states]
            link_orientations = [state[1] for state in link_states]

            # Generate a random index to select a link
            random_index = random.randint(3, len(link_positions) - 1)

            # Get the position and orientation of the randomly selected link
            selected_link = random_index
            selected_link_position = link_positions[random_index]
            selected_link_orientation = link_orientations[random_index]

            force = [400, 550, 700]  # Define the force vector (e.g., 10 N in the x-direction)
            position = [0, 0, 0]  # Define the position of the external force (e.g., at the joint's origin)
            p.applyExternalForce(robot, random_index, force, position, p.LINK_FRAME)  # Apply the force

            print("FORCE APPLIED AT LINK", random_index)
            print(f"FORCE APPLIED AT TIME: {current_time}")

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

        tau = np.dot(inertia_matrix, joint_accelerations) + np.dot(C, joint_velocities)

        # print(f"Torque: {tau}")

        last_residual = residuals[-1]

        integral_sum = tau + np.dot(np.transpose(C), joint_velocities) + np.array(last_residual)
        integral = integral_sum * current_time

        final_sum = p_t - integral - p_0

        new_residual = np.dot(gain_matrix, final_sum)

        residuals.append(new_residual)

        # print(f"New residual: {new_residual}")

        #####################################################

        # TODO: Stop the simulation based on residuals

        if current_time > 1.0:
            break

        # time.sleep(0.25)

    # Remove initial residual value
    residuals.pop(0)

    print(time_steps)

    index = time_steps.index(0.0875)

    time_steps = time_steps[index:]
    residuals = residuals[index:]
    all_momenta = all_momenta[index:]
    all_joint_velocities = all_joint_velocities[index:]
    kinetic_energy = kinetic_energy[index:]

    plot_graph("Plot of residuals over time",
               "residuals",
               "Residual",
               residuals,
               time_steps)

    plt.clf()

    plot_graph("Plot of momenta over time",
               "momenta",
               "Momentum (kg/rad s-2)",
               all_momenta,
               time_steps)

    plt.clf()

    plot_graph("Plot of velocity over time",
               "velocities",
               "Velocity (rad s-1)",
               all_joint_velocities,
               time_steps)

    plt.clf()

    plot_scalar_graph("Plot of kinetic energy over time",
                      "kinetic_energy",
                      "Kinetic Energy",
                      kinetic_energy,
                      time_steps)

    plt.clf()

    positions = all_link_positions[selected_link]
    positions = positions[index:]
    plot_3d_graph("Plot of collision-affected link movement",
                  "link_movement",
                  "Euclidean distance",
                  positions)

    p.disconnect()


def compute_coriolis_matrix(robot, joint_positions, joint_velocities, joint_accelerations):

    N = len(joint_positions)

    C = np.zeros(shape=(N, N))

    Csq = copy.copy(C)

    for i in range(N):

        QD = np.zeros(N)
        QD[i] = 1

        tau = p.calculateInverseDynamics(robot, list(joint_positions), list(joint_velocities),
                                         list(joint_accelerations))
        tau = np.array(tau)

        Csq[:, i] = Csq[:, i] + np.transpose(tau)

    for j in range(N):
        for k in range(j + 1, N):

            QD = np.zeros(N)
            QD[j] = 1
            QD[k] = 1

            tau = p.calculateInverseDynamics(robot, list(joint_positions), list(joint_velocities),
                                             list(joint_accelerations))
            tau = np.array(tau)

            product_term = np.transpose(tau) - Csq[:, k] - Csq[:, j]

            C[:, k] = C[:, k] + np.dot(product_term, joint_velocities[j])

    return C + Csq + np.diag(joint_velocities)


def plot_scalar_graph(title, figure_name, variable_name, variables, time_steps):

    timesteps = np.array(time_steps)

    plt.plot(timesteps, np.array(variables))

    plt.xlabel('Time (s)')
    plt.ylabel(variable_name)
    plt.title(title)
    plt.legend()

    plt.savefig(figure_name+".png")


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


def plot_3d_graph(title, figure_name, variable_name, variables):

    x = [point[0] for point in variables]
    y = [point[1] for point in variables]
    z = [point[2] for point in variables]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    highlight_index = collision_index
    ax.scatter(x[highlight_index], y[highlight_index], z[highlight_index], color='red', s=100)

    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_title(title)

    plt.savefig(figure_name+".png")


if __name__ == '__main__':
    main()
