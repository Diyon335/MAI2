import copy
import random

import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


robot_urdf = "kuka_lbr_iiwa_support/urdf/lbr_iiwa_14_r820.urdf"
collision_index = None


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

    duration=20
    time_step = 1/240
    p.setTimeStep(time_step)

    simulation_steps = int(duration/time_step)

    # Give the final orientation and joint positions of the robot
    final_orientation = p.getQuaternionFromEuler([3.14, 0, 0])
    final_joint_positions = p.calculateInverseKinematics(robot,
                                                         7,
                                                         [0.1, 0.1, 0.4],
                                                         targetOrientation=final_orientation,
                                                         maxNumIterations = simulation_steps
                                                         )
    
    print(final_joint_positions)

    change = []
    for jointIndex in range(num_joints-2):
        # targetPositions[jointIndex] = jointPositions[jointIndex]
        change.append(final_joint_positions[jointIndex] / duration)  # Calculate increment for each joint
    print(change)

    p_0 = None
    r_0 = None
    previous_velocities = None
    selected_link = None

    gain_values = [0.007] * 7
    gain_matrix = np.diag(gain_values)
    # print(gain_matrix)

    residuals = []
    time_steps = []
    all_joint_velocities = []
    all_momenta = []
    all_link_positions = {i: [] for i in range(num_joints)}
    kinetic_energy = []

    position_errors = []

    # Display in the simulation how the robot goes from initial position, to our desired position

    startTime = time.time()
    i = -1
    force_not_applied = True

    p.setRealTimeSimulation(0)

    while int(time.time() - startTime) < duration:
        i = i+1
        # print(i)
        
        p.stepSimulation()
        time.sleep(time_step)

        # if force_not_applied:
        curr_pos = [state[0] for state in p.getJointStates(robot, range(7))]
        move = [ele1+ele2 for ele1, ele2 in zip(curr_pos, change)]

        # Set all joints to the final desired position
        for jointIndex in range(num_joints):
            p.setJointMotorControlArray(robot, range(7), p.POSITION_CONTROL, move)

        current_time = time.time()-startTime
        # print(current_time)
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
        error = np.subtract(np.array(joint_positions), np.array(final_joint_positions))
        error = np.linalg.norm(error)
        position_errors.append(error)
        ################################################################

        inertia_matrix = np.array(p.calculateMassMatrix(robot, joint_positions))

        # print(f"Inertia matrix: {inertia_matrix}")
        # print(inertia_matrix.shape)

        #################################################################
        ke = np.linalg.multi_dot([np.array(joint_velocities), inertia_matrix,
                                  np.array(joint_velocities).reshape(-1, 1)])
        kinetic_energy.append(ke[0])

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

        # Calculate joint forces acting on the robot
        tau = p.calculateInverseDynamics(robot, list(joint_positions), list(joint_velocities),
                                                   list(joint_accelerations))

        gravity_vector = p.calculateInverseDynamics(robot, list(joint_positions), [0.0] * len(joint_positions), [0.0] *
                                                    len(joint_positions))

        #################################################################

        if current_time > 12 and force_not_applied:
            force_not_applied = False
            # Retrieve the link positions and orientations
            link_states = p.getLinkStates(robot, range(7))

            link_positions = [state[0] for state in link_states]
            link_orientations = [state[1] for state in link_states]

            # Generate a random index to select a link
            random_index = random.randint(3, len(link_positions) - 1)

            # Get the position and orientation of the randomly selected link
            selected_link = random_index
            selected_link_position = link_positions[selected_link]
            selected_link_orientation = link_orientations[selected_link]

            global collision_index
            collision_index = i

            # Create a visual marker at the selected point
            marker_size = 0.05
            marker_color = [1, 0, 0, 1]  # Red color
            marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=marker_size, rgbaColor=marker_color)
            marker_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=marker_size)

            marker_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker_visual,
                                            baseCollisionShapeIndex=marker_collision,
                                            basePosition=selected_link_position,
                                            baseOrientation=selected_link_orientation)

            # Run the simulation loop
            print("\n\n\n\n")
            print("FORCE APPLIED AT LINK", selected_link)
            print(f"FORCE APPLIED AT TIME: {current_time}")
            print("\n\n\n\n")

        ##################### Algorithm ################################
        if i == 0:
            continue

        last_residual = residuals[-1]

        integral_sum = np.add(np.array(tau), np.array(last_residual))
        integral_sum = np.subtract(integral_sum, gravity_vector)

        integral = integral_sum * current_time
        final_sum = np.subtract(p_t, integral)
        final_sum = np.subtract(final_sum, p_0)
        new_residual = np.dot(gain_matrix, final_sum)

        residuals.append(new_residual)
        threshold = 10
        mod_r = np.linalg.norm(new_residual)
        if mod_r > threshold:
            result_index = check_r(new_residual)
            if result_index is not None:
                print(new_residual, mod_r)
                print("Link :", result_index+1)
            else:
                print("No high number found with subsequent numbers closer to 0.")

        print(f"New residual: {new_residual}, mod_r: {mod_r}")

        #####################################################

    # Remove initial residual value
    residuals.pop(0)
    time_steps.pop(0)
    all_momenta.pop(0)
    all_joint_velocities.pop(0)
    kinetic_energy.pop(0)
    position_errors.pop(0)

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

    plot_scalar_graph("Plot of errors in joint position",
                      "error",
                      "Error",
                      position_errors,
                      time_steps)

    plt.clf()

    plot_scalar_graph("Plot of kinetic energy over time",
                      "kinetic_energy",
                      "Kinetic Energy",
                      kinetic_energy,
                      time_steps)

    plt.clf()

    positions = all_link_positions[selected_link]

    plot_3d_graph("Plot of collision-affected link movement",
                  f"link_movement_{selected_link}",
                  "Euclidean distance",
                  positions)

    p.disconnect()


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


def check_r(numbers):

    print(f"original: {numbers}")
    numbers = numbers[2:]
    print(f"modified: {numbers}")

    for i in range(len(numbers) - 1):

        next_values = numbers[i + 1:]

        if abs(np.mean(next_values)) < 1:
            return i+2

    # If no such index is found, return -1 or raise an exception
    return -1


if __name__ == '__main__':
    main()
