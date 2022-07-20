from collections import defaultdict, Counter
import numpy as np
import math
from helpers import wrap_angle

INFINITY = 10_000

def points_on_circumference(center, r, n=100):
    return [
        (
            round(center[0]+(math.cos(2 * np.pi / n * x) * r)),  # x
            round(center[1] + (math.sin(2 * np.pi / n * x) * r))  # y
        ) for x in range(0, n + 1)]

class VotingGrid:

    def __init__(self, M, x_min, x_max, y_min, y_max):
        self.M = M

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.grid = defaultdict(Counter)

    def vote(self, mu, z, label):

        # Obtain the hypothesis of the landmarks
        hypotheses = points_on_circumference(mu[:2], z)

        # Remove duplicates
        hypotheses = list(set(hypotheses))

        # Remove hypothesis that are outside of the grid
        hypotheses = [hyp for hyp in hypotheses
                     if self.x_min <= hyp[0] <= self.x_max and
                        self.y_min <= hyp[1] <= self.y_max]

        # Cast the votes
        self.grid[label] += Counter(hypotheses)

    def converged(self, label):
        
        # Obtain the two most common hypotheses
        most_common = self.grid[label].most_common(2)

        # Is there a clear winner?
        return most_common[0][1] >= 5

    def most_likely(self, label):
        return self.grid[label].most_common(1)[0][0]

class RangeOnlyEKFSLAM:

    def __init__(self, M, R, Q, dt=0.1):
        """
        A simple EKF-SLAM implementation using a Range-Only sensor
        with known data association

        Arguments:
        M     - The number of landmarks that we expect
        R     - The 3x3 covariance matrix of the motion noise
        Q     - The standard deviation of the measurement noise
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

        # Setup a voting grid
        self.voting_grid = VotingGrid(M, 0, 30, 0, 30)

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
        for (r, l) in z:

            l = int(l)

            # Vote for the current landmark
            self.voting_grid.vote(mu_prime[:2], r, l)

            # Continue if the voting grid has not converged
            if not self.voting_grid.converged(l):
                continue

            # Initialize the landmark
            if self.mu[3 + 2*l] == 0:
                mx, my = self.voting_grid.most_likely(l)
                mu_prime[3 + 2*l]     = mx
                mu_prime[3 + 2*l + 1] = my
                print(mx, my)

            # Compute useful helpers
            dx = mu_prime[0] - mu_prime[3 + 2*l]
            dy = mu_prime[1] - mu_prime[3 + 2*l + 1]
            q = dx**2 + dy**2
            
            # Compute the expected observation
            z_hat = q**0.5

            # # Setup the helper matrix
            F = np.zeros((5, 3 + self.M*2))
            F[:3, :3] = np.eye(3)
            F[3, 3 + 2*l]     = 1
            F[4, 3 + 2*l + 1] = 1

            # Setup the derivative of the state with respect to the observation
            H = 1/q *\
                np.array([[
                    dx, dy, 0, -dx, -dy
                ]], dtype='float')\
                .dot(F)
            
            # Compute the Kalman Gain
            K = sigma_prime.dot(H.T).dot(np.linalg.pinv(H.dot(sigma_prime).dot(H.T) + self.Q))

            # Compute the difference between expected and real observation
            z_diff = r - z_hat

            # Update the mean and covariance
            mu_prime    += K.dot(z_diff).flatten()
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