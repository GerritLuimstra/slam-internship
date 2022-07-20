import numpy as np
import math
import time
from scipy.stats import multivariate_normal
from .particle import Particle, weighted_choice
import copy

class FASTSLAM:

    def __init__(self, N, M, Q, R, P, dt=0.1):
        """
        A simple FASTSLAM implementation using a Range-Bearing sensor
        with known data association

        Arguments:
        N     - The number of particles to use in the system
        M     - The number of landmarks that we expect
        Q     - The 2x2 covariance matrix of the measurement noise
        R     - The 3x3 covariance matrix of the motion noise
        P     - The 3x3 covariance matrix of the particle noise
        dt    - The timestep
        """
        self.N = N
        self.M = M
        self.Q = Q
        self.R = R
        self.P = P
        self.dt = dt

        # Generate the initial particles
        self.particles = [Particle(
            # np.random.uniform(0, 30), 
            # np.random.uniform(0, 30), 
            # np.random.uniform(-np.pi, np.pi),
            2, 2, 0,
            self.Q,
            self.R,
            self.P
        ) for _ in range(N)]

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

    def predict_landmarks(self):
        """
        Predicts the positions of the landmarks
        """

        # Obtain the most likely particle
        most_likely = np.argmax([particle.weight for particle in self.particles])

        # Obtain its predictions with regards to the landmarks
        predictions = self.particles[most_likely].landmarks
        predicted_landmarks = [[0, 0] for _ in range(self.M)]
        for index, (mu, _) in predictions.items():
            predicted_landmarks[index] = mu
        
        return predicted_landmarks

    def most_likely_particle(self):
        return self.particles[np.argmax([particle.weight for particle in self.particles])]

    def predict_position(self):
        """
        Predicts the positions of the robot
        """
        most_likely = np.argmax([particle.weight for particle in self.particles])
        return self.particles[most_likely].x, self.particles[most_likely].y, self.particles[most_likely].theta




