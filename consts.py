import math

import numpy as np

LOWER_BOUNDARY_HSV = np.array([0, 0, 0])
UPPER_BOUNDARY_HSV = np.array([255, 255, 255])
DIS_CAMERA_TO_SHOOTER = 17  # cm
ANGLE_CAMERA_TO_ROBOT = 90  # deg
TARGET_AREA = 7.5  # squared cm
CIRC_TARGET_AREA = (1.3**2) * math.pi
FOV_HORIZONTAL = 55  # deg
FOV_VERTICAL = 55  # deg
VIDEO_WIDTH = 640  # px
VIDEO_HEIGHT = 480  # px
FOCAL_LENGTH = 573.7
a1FOCAL_LENGTH = 678.5  # cm
