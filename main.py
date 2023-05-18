import random

import pybullet as p
import time
import pybullet_data


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

    # Display in the simulation how the robot goes from initial position, to our desired position
    for i in range(100):
        print(i)
        p.stepSimulation()

        for joint_index in range(num_joints):
            dynamics_info = p.getDynamicsInfo(robot, joint_index)
            # print(dynamics_info)
            inertia_matrix = dynamics_info[2]
            # print(f"Inertia matrix for joint {joint_index}:")
            # print(inertia_matrix)

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
