import numpy as np
import math
import time
from scipy.stats import multivariate_normal
import copy

import random
import math
import numpy as np
from scipy.stats import multivariate_normal
from helpers import multi_normal, wrap_angle

def weighted_choice(choices, weights):
    total = sum(weights)
    if total == 0:
        return random.choice(choices)
    treshold = random.uniform(0, total)
    for k, weight in enumerate(weights):
        total -= weight
        if total < treshold:
            return choices[k]

INFINITY = 10_000
DT = 1

class Particle:

    def __init__(self, x, y, theta, L, P):
        """
        A particle that encodes an hypothesis of the state of the FASTSLAM algorithm

        Arguments:
        x     - The x-coordinate of the hypothesis
        y     - The y-coordinate of the hypothesis
        theta - The angle of the hypothesis
        L     - The list of landmarks in the environment
        P     - The 3x3 covariance matrix of the particle noise
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.L = L
        self.P = P

        self.weight = np.exp(-10)

    def step(self, u, z):
        """
        Updates the state of the particle based on the executed move and incoming observation

        Arguments:
        u     - The move that was issued by the robot
        z     - The observation(s) that the robot sensed after the move
        """

        # Unpack the move
        v, w = u

        # Move the position according to the motion model
        self.x += -v/w * math.sin(self.theta) + v/w * math.sin(self.theta + w*DT)
        self.y +=  v/w * math.cos(self.theta) - v/w * math.cos(self.theta + w*DT)
        self.theta += w*DT

        # Add a little noise
        self.x, self.y, self.theta = np.array(
            [self.x, self.y, self.theta]
        ) + np.random.multivariate_normal(np.zeros(3), self.P)

        # Wrap the angle
        self.theta = wrap_angle(self.theta)

        # Process the observations
        weight = 1 if len(z) > 0 else np.exp(-10)

        # Process the observations
        for (r, l) in z:
            l = int(l)

            # Compute the expected distance
            expected_distance = ((self.x - self.L[l][0])**2 + (self.y - self.L[l][1])**2)**0.5

            # Compute the weight of the observation
            obs_weight = 1 / (abs(r - expected_distance) + 1)**2

            # print(expected_distance, r)
            # print(obs_weight)

            weight *= obs_weight

        # Update the weight
        if len(z) > 0:
            self.weight = weight

    def __str__(self) -> str:
        return f"{self.x} {self.y} {self.theta}"

class WiFiParticleFilter:

    def __init__(self, N, L, Q, R, P, dt=0.1):
        """
        A simple FASTSLAM implementation using a Range-Bearing sensor
        with known data association

        Arguments:
        N     - The number of particles to use in the system
        L     - The list of landmarks in the environment
        Q     - The 2x2 covariance matrix of the measurement noise
        R     - The 3x3 covariance matrix of the motion noise
        P     - The 3x3 covariance matrix of the particle noise
        dt    - The timestep
        """
        self.N = N
        self.Q = Q
        self.R = R
        self.P = P
        self.dt = dt

        # Generate the initial particles
        self.particles = [Particle(
            np.random.uniform(0, 30), 
            np.random.uniform(0, 30), 
            np.random.uniform(-np.pi, np.pi),
            L,
            self.P
        ) for _ in range(N)]

        # self.particles = [
        #     Particle(
        #         3, 
        #         3,
        #         np.random.uniform(-np.pi, np.pi),
        #         L,
        #         self.P
        #     ), Particle(
        #         15, 15,
        #         np.random.uniform(-np.pi, np.pi),
        #         L,
        #         self.P
        #     )
        # ]

    def step(self, u, z):
        """
        Updates the state of the FASTSLAM based on the executed move and incoming observation

        Arguments:
        u     - The move that was issued by the robot
        z     - The observation(s) that the robot sensed after the move
        """
        
        # Move the particles, i.e. sample from the motion model
        for particle in self.particles:
            particle.step(u, z)

        # Obtain the weights
        # but only if there is an observation
        if len(z) > 0:
            
            # Obtain the particles weights
            weights = [particle.weight for particle in self.particles]

            # Resample the particles
            idxs = [weighted_choice(list(range(self.N)), weights) for _ in range(self.N)]

            # Update the particles
            self.particles = [copy.copy(self.particles[idx]) for idx in idxs]

    def predict_position(self):
        """
        Predicts the positions of the robot
        """
        most_likely = np.argmax([particle.weight for particle in self.particles])
        return self.particles[most_likely].x, self.particles[most_likely].y, self.particles[most_likely].theta




