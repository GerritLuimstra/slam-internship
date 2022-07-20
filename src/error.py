import numpy as np
import math

def measure_error(world, trajectory, slam, filter=False):

    # Keep track of the predicted path
    predicted_path = [trajectory.positions[0]]

    # Run until the user asks to quit
    running = True
    idx = 0
    while idx < len(trajectory.moves) and running:

        # Advance the slam
        slam.step(trajectory.moves[idx], trajectory.observations[idx+1])

        # Keep track of the predicted path
        predicted_path.append(slam.predict_position())

        idx += 1

    # Compute the position error
    mean_position_error = np.mean([math.dist(trajectory.positions[i][:2], predicted_path[i][:2]) for i in range(0, len(predicted_path))])

    if not filter:

        # Compute the landmark errors
        predicted_lms = slam.predict_landmarks()
        mean_landmark_errors = np.mean([math.dist(world.landmarks[i][:2], m) for i, m in enumerate(predicted_lms)])

    if filter:
        return mean_position_error

    return mean_position_error, mean_landmark_errors

