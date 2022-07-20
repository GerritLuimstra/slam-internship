import numpy as np
import math

def multi_normal(x, mean, cov):
    """Calculate the density for a multinormal distribution"""
    den = 2 * math.pi * math.sqrt(np.linalg.det(cov))
    num = np.exp(-0.5*np.transpose((x - mean)).dot(np.linalg.inv(cov)).dot(x - mean))
    result = num/den
    return result

def wrap_angle(angle):
    # ensures angle in [0, 2pi]
    if angle > 2*np.pi:
        angle = angle - 2*np.pi
    if angle < 0:
        angle = angle + 2*np.pi
    return angle
