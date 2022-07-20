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

    def __init__(self, x, y, theta, Q, R, P):
        """
        A particle that encodes an hypothesis of the state of the FASTSLAM algorithm

        Arguments:
        x     - The x-coordinate of the hypothesis
        y     - The y-coordinate of the hypothesis
        theta - The angle of the hypothesis
        Q     - The 2x2 covariance matrix of the measurement noise
        R     - The 3x3 covariance matrix of the motion noise
        P     - The 3x3 covariance matrix of the particle noise
        """
        self.x = x
        self.y = y
        self.theta = theta

        self.R = R
        self.Q = Q
        self.P = P

        self.landmarks = {}
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
        for (r, angle, l) in z:
            
            l = int(l)

            # Initialize landmark, if not already seen before
            if not l in self.landmarks:

                # Guess the landmark x, y
                lm_x = self.x + r * math.cos(angle + self.theta)
                lm_y = self.y + r * math.sin(angle + self.theta)

                # Compute the Jacobian and its inverse
                dx = lm_x - self.x
                dy = lm_y - self.y
                q = dx**2 + dy**2
                d = q ** 0.5
                H = np.array([[dx/d, dy/d], [-dy/q, dx/q]])
                H_inv = np.linalg.pinv(H)

                # Initialize the covariance
                sigma = H_inv.dot(self.Q).dot(H_inv.T)

                # Initialize the landmark
                self.landmarks[l] = (np.array([lm_x, lm_y]), sigma)

                # Compute the observation weight
                obs_weight = 1

                continue

            # Obtain the landmark information
            mu, sigma = self.landmarks[l]

            # Compute useful helpers
            dx = mu[0] - self.x
            dy = mu[1] - self.y
            q = dx**2 + dy**2
            d = q ** 0.5

            # Compute the expected observation
            z_hat = [d, np.arctan2(dy, dx) - self.theta]
            z_hat[1] = wrap_angle(z_hat[1])

            # Compute the Jacobian
            H = np.array([[dx/d, dy/d], [-dy/q, dx/q]])

            # Compute the measurement covariance
            Q = H.dot(sigma).dot(H.T) + self.Q

            # Compute the Kalman Gain
            K = sigma.dot(H.T).dot(np.linalg.pinv(Q))

            # Compute the difference between actual and expected observation
            z_diff = np.array([r, angle]) - z_hat
            z_diff[1] = (z_diff[1] + np.pi) % (2*np.pi) - np.pi

            # Update mu and sigma
            mu_prime = mu + K.dot(z_diff)
            sigma_prime = (np.identity(2) - K.dot(H)).dot(sigma)
            
            # Update the landmark
            self.landmarks[l] = (mu_prime, sigma_prime)

            # Compute the observation weight
            pdf = multivariate_normal([r, angle], Q)
            obs_weight = pdf.pdf(z_hat)
            
            weight *= obs_weight

        # Update the weight
        if len(z) > 0:
            self.weight = weight

    def __str__(self) -> str:
        return f"{self.x} {self.y} {self.theta}"