class WorldSettings:

    def __init__(self, Q, R, alpha, distance):
        self.Q = Q
        self.R = R
        self.alpha = alpha
        self.distance = distance

class World:

    def __init__(self, settings, landmarks):
        self.settings = settings
        self.landmarks = landmarks

