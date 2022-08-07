import numpy as np
from world.world import WorldSettings, World
from trajectory.generate import spiral_trajectory, square_trajectory
from kalman.extended_kalman_wifi import WiFiExtendedKalmanFilter
from kalman.extended_kalman_bearing import BearingExtendedKalmanFilter
from sensor import RangeBearingSensor, RangeOnlySensor
from visualize import *
from error import measure_error
from pprint import pprint
import tqdm
import sys

TRIES = 5

# Setup global options
np.set_printoptions(suppress=True, precision=3)
# np.random.seed(42)

# Setup the world settings
R = np.diag([0.1, 0.1, (5*np.pi)/180])

# Setup the landmarks
landmarks = [[np.random.uniform(0, 30), np.random.uniform(0, 30), int(i)] for i in range(20)]

noise_matrix = {}

for distance_noise in tqdm.tqdm([1, 2, 3, 4, 5, 6, 7]):
    for angle_noise in tqdm.tqdm([1, 2, 3, 4, 5, 6, 7]):

        measurement_errors = []
        for _ in range(TRIES):

            Q = np.array([
                [distance_noise, 0], [0, (angle_noise*np.pi)/180]
            ])

            alpha = np.pi
            max_distance = np.inf
            settings = WorldSettings(Q, R, alpha, max_distance)
            world = World(settings, landmarks)

            if sys.argv[1] == "range-only":

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

            # Measure the error
            mean_position_error = measure_error(world, trajectory, filter, filter=True)
            measurement_errors.append(mean_position_error)

        noise_matrix[(distance_noise, angle_noise)] = np.mean(np.array(measurement_errors))

pprint(noise_matrix)