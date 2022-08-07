import numpy as np
import math
from helpers import wrap_angle_pi

class RangeBearingSensor:

    def __init__(self, landmarks, Q, alpha, max_range):
        self.landmarks = landmarks
        self.Q = Q
        self.alpha = alpha
        self.max_range = max_range

    def sense(self, mu):

        x, y, theta = mu[:3]
        
        Z = []

        for i in range(len(self.landmarks)):
            
            # Compute the range and bearing measurements
            z = np.zeros(3)
            z[0] = math.sqrt((x - self.landmarks[i][0])**2 + (y - self.landmarks[i][1])**2)
            z[1] = np.arctan2(self.landmarks[i][1] - y, self.landmarks[i][0] - x) - theta

            # Add sensor noise
            z[:2] += np.random.multivariate_normal(np.zeros(2), self.Q)
            
            # Add the label
            z[2] = self.landmarks[i][2]

            # Wrap relative bearing
            z[1] = wrap_angle_pi(z[1])

            # Only add the measurement if it is in the field of view
            # and within range
            if np.abs(z[1]) < self.alpha/2 and z[0] < self.max_range:
                Z.append(z)

        return Z

class RangeOnlySensor:

    def __init__(self, landmarks, Q, max_range):
        self.landmarks = landmarks
        self.Q = Q
        self.max_range = max_range

    def sense(self, mu):

        x, y, _ = mu[:3]
        
        Z = []

        for i in range(len(self.landmarks)):
            
            z = np.zeros(2)

            # Compute the range measurement
            z[0] = math.sqrt((x - self.landmarks[i][0])**2 + (y - self.landmarks[i][1])**2)

            # Add sensor noise
            z[0] += np.random.normal(loc=0, scale=self.Q)
            
            # Add the label
            z[1] = self.landmarks[i][2]

            # Only add the measurement if it is within range
            if z[0] < self.max_range:
                Z.append(z)

        return Z