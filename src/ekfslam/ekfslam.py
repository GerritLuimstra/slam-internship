import numpy as np
import math
from helpers import wrap_angle

INFINITY = 10_000

class EKFSLAM:

    def __init__(self, M, R, Q, dt=0.1):
        """
        A simple EKF-SLAM implementation using a Range-Bearing sensor
        with known data association

        Arguments:
        M     - The number of landmarks that we expect
        R     - The 3x3 covariance matrix of the motion noise
        Q     - The 2x2 covariance matrix of the measurement noise
        dt    - The timestep
        """
        self.M = M
        self.R = R
        self.Q = Q
        self.dt = dt

        # Initialize mu and sigma
        self.mu    = np.zeros(M*2 + 3)
        self.sigma = np.eye(M*2 + 3)*INFINITY
        np.fill_diagonal(self.sigma[:3, :3], 0)

    def set_state(self, mu):
        """
        Sets the state to the new mu

        Arguments:
        mu - The state to update to
        """
        self.mu = mu

    def step(self, u, z):
        """
        Updates the state of the EKF-SLAM based on the executed move and incoming observation

        Arguments:
        u     - The move that was issued by the robot
        z     - The observation(s) that the robot sensed after the move
        """

        # Extract relevant information
        v, w, = u

        # Setup the helper matrix
        F = np.zeros((3, 3 + 2*self.M))
        np.fill_diagonal(F, 1)

        # Compute the derivative
        K = np.zeros((3, 3))
        K[0, 2] = -v/w * math.cos(self.mu[2]) + v/w * math.cos(self.mu[2] + w*self.dt)
        K[1, 2] = -v/w * math.sin(self.mu[2]) + v/w * math.sin(self.mu[2] + w*self.dt)
        # K += np.eye(3)

        # Perform the prediction step
        mu_prime      = self.mu.copy()
        mu_prime[:3] += [
            -v/w * math.sin(self.mu[2]) + v/w * math.sin(self.mu[2] + w*self.dt),
            v/w * math.cos(self.mu[2]) - v/w * math.cos(self.mu[2] + w*self.dt),
            w*self.dt
        ]
        mu_prime[2] = wrap_angle(mu_prime[2])
        G = np.eye(mu_prime.shape[0]) + F.T.dot(K).dot(F)
        sigma_prime = G.dot(self.sigma).dot(G.T) + F.T.dot(self.R).dot(F)

        # Loop over the observations and perform the correction step
        for (r, angle, l) in z:

            l = int(l)

            # Initialize a landmark, if we haven't seen it before
            if self.sigma[3 + 2*l, 3 + 2*l] == INFINITY:
                mu_prime[3 + 2*l]     = mu_prime[0] + r * math.cos(angle + mu_prime[2])
                mu_prime[3 + 2*l + 1] = mu_prime[1] + r * math.sin(angle + mu_prime[2])

            # Compute useful helpers
            dx = mu_prime[3 + 2*l] - mu_prime[0]
            dy = mu_prime[3 + 2*l + 1] - mu_prime[1]
            q = dx**2 + dy**2
            
            # Compute the expected observation
            z_hat = [q**0.5, np.arctan2(dy, dx) - mu_prime[2]]
            z_hat[1] = wrap_angle(z_hat[1])

            # Setup the helper matrix
            F = np.zeros((5, 3 + self.M*2))
            F[:3, :3] = np.eye(3)
            F[3, 3 + 2*l]     = 1
            F[4, 3 + 2*l + 1] = 1

            # Setup the derivative of the state with respect to the observation
            H = 1/q *\
                np.array([[-q**0.5 *dx, -q**0.5*dy, 0, q**0.5*dx, q**0.5*dy],
                                [dy, -dx, -q, -dy, dx]], dtype='float')\
                .dot(F)
            
            # Compute the Kalman Gain
            K = sigma_prime.dot(H.T).dot(np.linalg.pinv(H.dot(sigma_prime).dot(H.T) + self.Q))

            # Compute the difference between expected and real observation
            z_diff = np.array([r, angle]) - z_hat
            z_diff[1] = (z_diff[1] + np.pi) % (2*np.pi) - np.pi

            # Update the mean and covariance
            mu_prime    += K.dot(z_diff)
            sigma_prime = (np.eye(3 + self.M*2) - K.dot(H)).dot(sigma_prime)

        # Update the state
        self.mu = mu_prime
        self.sigma = sigma_prime

    def predict_landmarks(self):
        """
        Predicts the positions of the landmarks
        """
        lm_positions = self.mu[3:]
        return [lm_positions[offs:offs+2] for offs in range(0, len(lm_positions), 2)]

    def predict_position(self):
        """
        Predicts the positions of the robot
        """
        return self.mu[:2]