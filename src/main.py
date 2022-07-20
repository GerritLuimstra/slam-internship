import numpy as np
from world.world import WorldSettings, World
from trajectory.generate import spiral_trajectory, square_trajectory
from ekfslam.ekfslam import EKFSLAM
from ekfslam.roekfslam import RangeOnlyEKFSLAM
from fastslam.fastslam import FASTSLAM
from sensor import RangeBearingSensor, RangeOnlySensor
from visualize import *
from error import measure_error
import sys
from pprint import pprint

# Setup global options
np.set_printoptions(suppress=True, precision=3)
np.random.seed(42)

# Setup the world settings
R = np.diag([0, 0, np.deg2rad(1)])**2
Q = np.array([[0, 0],
              [0, 0]])
P = np.diag([0, 0, np.deg2rad(0)])
alpha = np.pi/2
max_distance = 30
settings = WorldSettings(Q, R, alpha, max_distance)

# Setup the world
landmarks = [[np.random.uniform(0, 30), np.random.uniform(0, 30), int(i)] for i in range(3)]

world = World(settings, landmarks)

if sys.argv[4] == "range-only":

    # Overwrite the range standard deviation
    Q = 2

    # Setup the sensor
    sensor = RangeOnlySensor(landmarks, Q, max_distance)

else:

    # Setup the sensor
    sensor = RangeBearingSensor(landmarks, Q, alpha, max_distance)

# Obtain a trajectory
trajectory = square_trajectory(sensor, Q, add_noise=False)\
            if sys.argv[3] == "square" else\
            spiral_trajectory(sensor, Q, add_noise=False)

if sys.argv[2] == "ekf":

    # Initialize the state of the SLAM
    mu    = np.zeros(len(landmarks)*2 + 3)
    mu[0] = trajectory.positions[0][0]
    mu[1] = trajectory.positions[0][1]
    mu[2] = trajectory.positions[0][2]

    if sys.argv[4] == "range-only":

        # Setup the SLAM object
        slam = RangeOnlyEKFSLAM(len(landmarks), R, Q, dt=1)
    else:
        # Setup the SLAM object
        slam = EKFSLAM(len(landmarks), R, Q, dt=1)

    # Initialize the slam state
    slam.set_state(mu)

else:

    # Setup the SLAM object
    slam = FASTSLAM(5, len(landmarks), Q, R, P, dt=1)

if sys.argv[1] == "visualize":

    # Visualize the progress
    visualize_world(world, trajectory, slam, range_only=True)

else:
    
    # Measure the error
    mean_position_error, mean_landmark_error = measure_error(world, trajectory, slam)

    print(mean_position_error, mean_landmark_error)