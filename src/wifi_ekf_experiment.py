import numpy as np
from world.world import WorldSettings, World
from trajectory.generate import spiral_trajectory, square_trajectory
from kalman.extended_kalman_wifi import WiFiExtendedKalmanFilter
from kalman.extended_kalman_bearing import BearingExtendedKalmanFilter
from sensor import RangeBearingSensor, RangeOnlySensor
from visualize import *
from error import measure_error
import sys

# Setup global options
np.set_printoptions(suppress=True, precision=3)
np.random.seed(42)

# Setup the world settings
R = np.diag([0.1, 0.1, 0])**2

if sys.argv[2] == "range-only":
    Q = 1
else:
    Q = np.array([
        [1, 0], [0, 0.01]
    ])

alpha = np.pi
max_distance = np.inf
settings = WorldSettings(Q, R, alpha, max_distance)

# Setup the world
landmarks = [[np.random.uniform(0, 30), np.random.uniform(0, 30), int(i)] for i in range(5)]
world = World(settings, landmarks)

if sys.argv[2] == "range-only":

    # Setup the sensor
    sensor = RangeOnlySensor(landmarks, Q, max_distance)

    # Setup the filter
    filter = WiFiExtendedKalmanFilter(landmarks, R, Q, dt=1)

else:

    # Setup the sensor
    sensor = RangeBearingSensor(landmarks, Q, alpha, max_distance)

    # Setup the filter
    filter = BearingExtendedKalmanFilter(landmarks, R, Q, dt=1)

# Obtain a trajectory
trajectory = square_trajectory(sensor, None, add_noise=False)

# Initialize the state of the SLAM
mu    = np.zeros(3)
mu[0] = trajectory.positions[0][0]
mu[1] = trajectory.positions[0][1]
mu[2] = trajectory.positions[0][2]
filter.set_state(mu)

if sys.argv[1] == "visualize":

    # Visualize the progress
    visualize_world(world, trajectory, filter, range_only=True, filter=True)

else:
    
    # Measure the error
    mean_position_error = measure_error(world, trajectory, filter, filter=True)

    print(mean_position_error)