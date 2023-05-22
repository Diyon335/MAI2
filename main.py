import random

import pybullet as p
import time
import pybullet_data
import numpy as np
import sympy as sp


robot_urdf = "kuka_lbr_iiwa_support/urdf/lbr_iiwa_14_r820.urdf"


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

    prev_joint_velocities = [0.0] * num_joints

    # Display in the simulation how the robot goes from initial position, to our desired position
    for i in range(100):
        print(i)
        p.stepSimulation()

        # Retrieve joint positions and velocities
        joint_positions = []
        joint_velocities = []
        for joint_index in range(num_joints):
            joint_state = p.getJointState(robot, joint_index)
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])

        print(f"Joint positions: {joint_positions}")
        print(f"Joint velocities: {joint_velocities}")
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
        joint_velocities = np.transpose(joint_velocities)

        ################################################################

        # Compute p(t) as M * q_dot
        print("Calculating momentum")
        p_t = np.dot(inertia_matrices, joint_velocities)

        print(f"p(t): {p_t}")
        ################################################################

        # # Compute the Coriolis matrix
        # print("Computing coriolis matrix")
        # zero_accelerations = np.zeros(num_joints)
        # coriolis_matrix = p.calculateInverseDynamics(robot, joint_positions, joint_velocities, zero_accelerations)

        # print(f"Coriolis matrix: {coriolis_matrix}")
        ################################################################
        # # Compute the gravity matrix
        # zero_velocities = np.zeros(num_joints)
        # print("Computing gravity matrix")
        # gravity_matrix = p.calculateInverseDynamics(robot, joint_positions, zero_velocities, zero_accelerations)
        #
        # print(f"Gravity matrix: {gravity_matrix}")
        ################################################################
        # Compute joint accelerations using finite difference method
        dt = 1.0 / 240.0  # Time step
        joint_accelerations = [(v - prev_v) / dt for v, prev_v in zip(joint_velocities, prev_joint_velocities)]

        joint_accelerations = np.transpose(joint_accelerations)

        print(f"Joint accelerations: {joint_accelerations}")

        prev_joint_velocities = joint_velocities

        ################################################################
        # Define symbolic variables
        q = sp.symbols('q:{}'.format(num_joints))  # Joint positions
        qd = sp.symbols('qd:{}'.format(num_joints))  # Joint velocities
        qdd = sp.symbols('qdd:{}'.format(num_joints))  # Joint accelerations

        # Compute the Coriolis and centrifugal terms symbolically
        C = sp.Matrix([sp.diff(inertia_matrices, qd[j]).dot(qd) for j in range(num_joints)]) / 2

        # Substitute the numerical values for joint positions and velocities
        C = C.subs(list(zip(q, joint_positions))).subs(list(zip(qd, joint_velocities)))

        # Evaluate the Coriolis and centrifugal terms
        coriolisCentrifugal = np.array(C.evalf()).astype(float)

        ################################################################

        total_torque = np.dot(inertia_matrices, joint_accelerations) + np.dot(C, joint_velocities) + gravity_matrix

        print(f"total torque: {total_torque}")

        print("---------------------------------------------")

        # This is just a placeholder to generate a collision at a random link of the robot
        if i == 20:
            # Retrieve the link positions and orientations
            link_states = p.getLinkStates(robot, range(p.getNumJoints(robot)))

            link_positions = [state[0] for state in link_states]
            link_orientations = [state[1] for state in link_states]

            # Generate a random index to select a link
            random_index = random.randint(0, len(link_positions))

            # Get the position and orientation of the randomly selected link
            selected_link_position = link_positions[random_index]
            selected_link_orientation = link_orientations[random_index]

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
