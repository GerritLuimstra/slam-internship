import numpy as np
import math
import pygame
import pygame.gfxdraw
from math import atan2, cos, degrees, radians, sin
from ekfslam.ekfslam import INFINITY

TPS = 5
DOT_SIZE = 5
PRED_COLOR = (255, 0, 0)
PARTICLE_COLOR = (255, 140, 0)
ACTUAL_COLOR = (0, 255, 0)
LM_COLOR = (0, 0, 255)
COORDINATE_SIZE = 500//30

def coord_to_screen(x, y):
    return round(x * COORDINATE_SIZE + 55), round(y * COORDINATE_SIZE + 55)

def draw_thick_line(surface, point1, point2, thickness, color):

    def move(rotation, steps, position):
        """Return coordinate position of an amount of steps in a direction."""
        xPosition = cos(radians(rotation)) * steps + position[0]
        yPosition = sin(radians(rotation)) * steps + position[1]
        return (xPosition, yPosition)
    
    angle = degrees(atan2(point1[1] - point2[1], point1[0] - point2[0]))

    vertices = list()
    vertices.append(move(angle-90, thickness, point1))
    vertices.append(move(angle+90, thickness, point1))
    vertices.append(move(angle+90, thickness, point2))
    vertices.append(move(angle-90, thickness, point2))

    pygame.gfxdraw.aapolygon(surface, vertices, color)
    pygame.gfxdraw.filled_polygon(surface, vertices, color)

def visualize_world(world, trajectory, slam, range_only=False, filter=False):
    
    # Initialize the game
    pygame.init()

    # Set up the drawing window
    screen = pygame.display.set_mode([600, 600])

    # Keep track of the predicted path
    predicted_path = [trajectory.positions[0]]

    # Run until the user asks to quit
    running = True
    idx = 0
    while idx < len(trajectory.moves) and running:

        # Did the user click the window close button?
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Fill the background with white
        screen.fill((255, 255, 255))

        # Display the algorithm name
        font = pygame.font.SysFont(None, 24)
        img = font.render("EKFSLAM" if "sigma" in dir(slam) else "FastSLAM", True, (0, 0, 0))
        screen.blit(img, (220, 0))

        # Draw the landmarks
        for i, landmark in enumerate(world.landmarks):

            # Display the landmark normally
            x, y = coord_to_screen(landmark[0], landmark[1])

            if "sigma" in dir(slam) and not filter:

                # if range_only:
                #     sensor_range = COORDINATE_SIZE*(round(world.settings.distance) + 1) 
                #     pygame.draw.ellipse(
                #         screen,
                #         (128, 200, 128), 
                #         (x - sensor_range//2, y - sensor_range//2, sensor_range, sensor_range)
                #     )

                # Visualize the uncertainty
                if slam.mu[3 + i*2] == 0:
                    font = pygame.font.SysFont(None, 24)
                    img = font.render("X", True, (255, 0, 0))
                    screen.blit(img, (x, y))
                else:
                    pred_x, pred_y = coord_to_screen(slam.mu[3 + i*2], slam.mu[3 + i*2 + 1])
                    x_uncertainty = 5#COORDINATE_SIZE*(round(slam.sigma[3 + i*2][3 + i*2]) + 1)
                    y_uncertainty = 5#COORDINATE_SIZE*(round(slam.sigma[3 + i*2+1][3 + i*2 + 1]) + 1)
                    pygame.draw.ellipse(
                        screen, 
                        (128, 128, 128), 
                        (pred_x - x_uncertainty//2, pred_y - y_uncertainty//2, x_uncertainty, y_uncertainty)
                    )

            if "most_likely_particle" in dir(slam):
                for (mu, sigma) in slam.most_likely_particle().landmarks.values():
                    pred_x, pred_y = coord_to_screen(mu[0], mu[1])
                    x_uncertainty = COORDINATE_SIZE*(round(sigma[0][0]) + 1)
                    y_uncertainty = COORDINATE_SIZE*(round(sigma[1][1]) + 1)
                    pygame.draw.ellipse(
                        screen,
                        (128, 128, 128), 
                        (pred_x - x_uncertainty//2, pred_y - y_uncertainty//2, x_uncertainty, y_uncertainty)
                    )
            
            # Display the landmark normally
            pygame.draw.circle(screen, LM_COLOR, (x, y), DOT_SIZE)
            font = pygame.font.SysFont(None, 24)
            img = font.render(str(i), True, (255, 0, 0))
            screen.blit(img, (x, y))

        # Visualize the paths
        for j in range(1, idx+1):
            
            # Visualize the actual path
            prev_x, prev_y = coord_to_screen(trajectory.positions[j-1][0], trajectory.positions[j-1][1])
            x, y = coord_to_screen(trajectory.positions[j][0], trajectory.positions[j][1])
            draw_thick_line(screen, (prev_x, prev_y), (x, y), 2, ACTUAL_COLOR)

            # Visualize the predicted path
            prev_x, prev_y = coord_to_screen(predicted_path[j-1][0], predicted_path[j-1][1])
            x, y = coord_to_screen(predicted_path[j][0], predicted_path[j][1])
            draw_thick_line(screen, (prev_x, prev_y), (x, y), 2, PRED_COLOR)

        if "sigma" in dir(slam):
            # Visualize the robot uncertainty
            pred_x,  pred_y = coord_to_screen(slam.mu[0], slam.mu[1])
            x_uncertainty, y_uncertainty = COORDINATE_SIZE*(round(slam.sigma[0][0]) + 1), COORDINATE_SIZE*(round(slam.sigma[1][1])+1)
            pygame.draw.circle(screen, PRED_COLOR, (pred_x,  pred_y), DOT_SIZE+2)
            pygame.draw.ellipse(screen, (128, 128, 128), (pred_x - x_uncertainty//2, pred_y - y_uncertainty//2, x_uncertainty, y_uncertainty))

        if "particles" in dir(slam):

            # Plot the particles
            for particle in slam.particles:
                part_x,  part_y = coord_to_screen(particle.x, particle.y)
                pygame.draw.circle(screen, PARTICLE_COLOR, (part_x,  part_y), DOT_SIZE)

        # Visualize the robot position
        robot_x, robot_y = coord_to_screen(trajectory.positions[idx][0], trajectory.positions[idx][1])
        pygame.draw.circle(screen, ACTUAL_COLOR, (robot_x, robot_y), DOT_SIZE+2)

        # Visualize the field of view
        if not range_only:
            fov_length = 50 if world.settings.distance == np.inf else world.settings.distance
            robot_x, robot_y = trajectory.positions[idx][0], trajectory.positions[idx][1]
            line1_p1_x, line1_p1_y = coord_to_screen(robot_x, robot_y)
            line1_p2_x, line1_p2_y = coord_to_screen(robot_x + fov_length*np.cos(trajectory.positions[idx][2] + world.settings.alpha/2), 
                                                    robot_y + fov_length*np.sin(trajectory.positions[idx][2] + world.settings.alpha/2))
            line2_p1_x, line2_p1_y = coord_to_screen(robot_x, robot_y)
            line2_p2_x, line2_p2_y = coord_to_screen(robot_x + fov_length*np.cos(trajectory.positions[idx][2] - world.settings.alpha/2), 
                                                    robot_y + fov_length*np.sin(trajectory.positions[idx][2] - world.settings.alpha/2))
            draw_thick_line(screen, (line1_p1_x, line1_p1_y), (line1_p2_x, line1_p2_y), 1, (0, 0, 255))
            draw_thick_line(screen, (line2_p1_x, line2_p1_y), (line2_p2_x, line2_p2_y), 1, (0, 0, 255))

        # Advance the slam
        slam.step(trajectory.moves[idx], trajectory.observations[idx+1])

        # Keep track of the predicted path
        predicted_path.append(slam.predict_position())


        if not filter:
            # Compute the landmark errors
            predicted_lms = slam.predict_landmarks()
            landmark_errors = np.mean([math.dist(world.landmarks[i][:2], m) for i, m in enumerate(predicted_lms) if not (m[0] == 0 and m[1] == 0)])

        # Display the mean errors
        mean_position_error = np.mean([math.dist(trajectory.positions[i][:2], predicted_path[i][:2]) for i in range(0, idx+1)])
        
        if not filter:
            mean_landmark_error = landmark_errors

        font = pygame.font.SysFont(None, 24)
        img = font.render("Mean position error: " + str(round(mean_position_error, 3)), True, (0, 0, 0))
        screen.blit(img, (380, 0))

        if not filter:

            font = pygame.font.SysFont(None, 24)
            img = font.render("Mean landmark error: " + str(round(mean_landmark_error, 3)), True, (0, 0, 0))
            screen.blit(img, (380, 20))

        pygame.display.flip()
        pygame.time.wait(1000//TPS)

        idx += 1

    pygame.quit()