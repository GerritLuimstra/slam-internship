import numpy as np
import math
import matplotlib.pyplot as plt
from trajectory.trajectory import Trajectory
from helpers import wrap_angle

def motion(mu, u, dt=1):
    theta = mu[2]
    v, w = u
    return np.array([
        -v/w * math.sin(theta) + v/w * math.sin(theta + w*dt),
        v/w * math.cos(theta) - v/w * math.cos(theta + w*dt),
        w*dt
    ])

def square_trajectory(sensor, Q, add_noise=True):
    positions = [
        np.array([2, 2, 0.0])
    ]
    observations = [
        sensor.sense(positions[0])
    ]

    # Setup the trajectory
    moves = [np.array([2, 0.00001])]
    moves = moves * 13
    moves.append(np.array([0, np.pi/2]))
    moves = moves * 10

    for move in moves:

        # Obtain the last position
        position = positions[-1].copy()

        # Execute the move
        # and potentially add a little noise to the move
        if add_noise:
            position += motion(position, move + np.random.multivariate_normal(np.zeros(2), Q))
        else:
            position += motion(position, move)

        # Wrap the angle
        position[2] = wrap_angle(position[2])

        # Record the trajectory along with the observations
        observations.append(sensor.sense(position))
        positions.append(position)
    
    return Trajectory(positions, moves, observations)

def spiral_trajectory(sensor, Q, add_noise=True):
    positions = [
        np.array([15, 15, 0.0])
    ]
    observations = [
        sensor.sense(positions[0])
    ]

    # Setup the trajectory
    moves = []
    for i in range(100):
        moves.append(np.array([i/25, np.pi/4]))

    for move in moves:

        # Add a little noise to the move
        if add_noise:
            m = move + np.random.multivariate_normal(np.zeros(2), Q)
        else:
            m = move

        # Obtain the last position
        position = positions[-1].copy()

        # Execute the move
        position += motion(position, m)

        # Record the trajectory along with the observations
        observations.append(sensor.sense(position))
        positions.append(position)

    return Trajectory(positions, moves, observations)

def visualize_trajectory(positions, landmarks, sensor):

    with plt.ion():
        
        # Setup figure and subplots
        ax1 = plt.subplot()

        for i, position in enumerate(positions):

            # Reset axis
            ax1.clear()
            ax1.set_xlim(0, 30)
            ax1.set_ylim(0, 30)

            # Plot landmarks and estimates
            for _, landmark in enumerate(landmarks):
                ax1.plot([landmark[0]], [landmark[1]], color="blue", marker="o")

            # Plot the robot
            ax1.plot(position[0], position[1], marker=(3, 0, position[2]*180/np.pi), markersize=5, linestyle='None')

            # Plot the Field of View
            ax1.plot([position[0], position[0]+50*np.cos(position[2] + sensor.alpha/2)], [position[1], position[1]+50*np.sin(position[2] + sensor.alpha/2)], color="r")
            ax1.plot([position[0], position[0]+50*np.cos(position[2] - sensor.alpha/2)], [position[1], position[1]+50*np.sin(position[2] - sensor.alpha/2)], color="r")

            # Visualize the paths
            actual_path = positions[:i+1]
            for i in range(1, len(actual_path)):
                ax1.plot([actual_path[i-1][0], actual_path[i][0]], [actual_path[i-1][1], actual_path[i][1]], c="green", label="Actual")

            # Set appropriate titles
            ax1.title.set_text("Trajectory")

            plt.tight_layout()
            plt.pause(0.1)