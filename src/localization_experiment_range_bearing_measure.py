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
import matplotlib.pyplot as plt

TRIES = 1

# Setup global options
np.set_printoptions(suppress=True, precision=3)
np.random.seed(42)

# Setup the world settings
R = np.diag([0.1, 0.1, (5*np.pi)/180])

# Setup the landmarks
landmarks = [[np.random.uniform(0, 30), np.random.uniform(0, 30), int(i)] for i in range(5)]

distance_noises = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
angle_noises = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
noise_matrix = np.zeros((len(distance_noises), len(angle_noises)))

for d_idx, distance_noise in tqdm.tqdm(enumerate(distance_noises)):
    for a_idx, angle_noise in tqdm.tqdm(enumerate(angle_noises)):

        measurement_errors = []
        for _ in range(TRIES):

            Q = np.array([
                [distance_noise, 0], [0, (angle_noise*np.pi)/180]
            ])

            alpha = np.pi
            max_distance = np.inf
            settings = WorldSettings(Q, R, alpha, max_distance)
            world = World(settings, landmarks)

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

        noise_matrix[d_idx, a_idx] = np.mean(np.array(measurement_errors))


plt.title("Noise Matrix for Range-Only Localization")
plt.imshow(noise_matrix, cmap='Greys')
plt.colorbar(fraction=0.03, pad=0.05)
plt.xlabel("Angle Noise in Degrees")
plt.ylabel("Distance noise")
for j in range(len(distance_noises)):
    for i in range(len(angle_noises)):
        c = noise_matrix[j,i]
        plt.text(i, j, str(round(c, 2)), va='center', ha='center')
plt.show()