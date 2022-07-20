import numpy as np
import math
from helpers import wrap_angle

class BearingExtendedKalmanFilter:

    def __init__(self, L, R, Q, dt=0.1):
        """
        A simple EKF implementation using a Range-Bearing sensor with known data association

        Arguments:
        L       - The landmarks of the world
        R       - The 3x3 covariance matrix of the motion noise
        Q       - The 2x2 covariance matrix of the measurement noise
        dt      - The timestep
        """
        self.L = L
        self.R = R
        self.Q = Q
        self.dt = dt

        # Initialize mu and sigma
        self.mu    = np.zeros(3)
        self.sigma = np.zeros((3,3))

    def set_state(self, mu):
        """
        Sets the state to the new mu

        Arguments:
        mu - The state to update to
        """
        self.mu = mu

    def g(self, v, w):
        return self.mu + [
            -v/w * math.sin(self.mu[2]) + v/w * math.sin(self.mu[2] + w*self.dt),
            v/w * math.cos(self.mu[2]) - v/w * math.cos(self.mu[2] + w*self.dt),
            w*self.dt
        ]

    def G(self, v, w):
        G_ = np.eye(3)
        G_[0][2] = -v/w * math.cos(self.mu[2]) + v/w * math.cos(self.mu[2] + w*self.dt)
        G_[1][2] = -v/w * math.sin(self.mu[2]) + v/w * math.sin(self.mu[2] + w*self.dt)
        return G_

    def step(self, u, z):
        """
        Updates the state of the EKF-SLAM based on the executed move and incoming observation

        Arguments:
        u     - The move that was issued by the robot
        z     - The observation(s) that the robot sensed after the move
        """

        # Extract relevant information
        v, w, = u

        # Perform the prediction step
        mu_prime      = self.g(v, w)
        mu_prime[2] = wrap_angle(mu_prime[2])
        G_ = self.G(v, w)
        sigma_prime = G_.dot(self.sigma).dot(G_.T) + self.R

        # Loop over the observations and perform the correction step
        for (r, angle, l) in z:

            l = int(l)

            # Compute useful helpers
            dx = self.L[l][0] - mu_prime[0]
            dy = self.L[l][1] - mu_prime[1]
            q = dx**2 + dy**2
            
            # Compute the expected observation
            z_hat = [q**0.5, np.arctan2(dy, dx) - mu_prime[2]]
            z_hat[1] = wrap_angle(z_hat[1])

            # Compute the derivative of the measurement
            # with respect to the robot position
            H = 1/q *\
                np.array([[-q**0.5*dx, -q**0.5*dy, 0],
                                [dy, -dx, -q]], dtype='float')

            # Correction step
            inv = np.linalg.pinv(H.dot(sigma_prime.dot(H.T)) + self.Q)
            K = sigma_prime.dot(H.T).dot(inv)

            # Update the mean and covariance
            mu_prime    += K.dot(np.array([r, angle]) - z_hat)
            sigma_prime = (np.eye(3) - K.dot(H)).dot(sigma_prime)

        # Update the state
        self.mu = mu_prime
        self.sigma = sigma_prime

    def predict_position(self):
        """
        Predicts the positions of the robot
        """
        return self.mu[:2]