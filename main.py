
import cv2
import numpy as np
from typing import List
from drum import Drum

CIRCLE_EDGE_DETECTION_THRESHOLD = 100 # higher value finds cleaner edges
CIRCLE_DETECTION_THRESHOLD = 200 # increase reduces false positives
BLUR_FACTOR = 0.5

def init_drums(img_path: str) -> List[Drum]:
    """
    Initializes Drum objects based on detected circles in an image.

    Params: 
        img_path (str): The file path to the image (e.g., 'image.png').

    Returns:
        drums (List[Drum]): A list of Drum objects, each with attributes 
        (x, y, radius) corresponding to a detected circle. 
        The 'sound' attribute is initialized to None.
    """
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    max_radius = min(width, height)
    kernel_size = int(min(height, width) * BLUR_FACTOR)
    kernel_size += 1 if kernel_size % 2 == 0 else 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 2)

    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=30, 
        param1=CIRCLE_EDGE_DETECTION_THRESHOLD, 
        param2=CIRCLE_DETECTION_THRESHOLD, 
        minRadius=10, 
        maxRadius=max_radius
    )

    # edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
    # success = cv2.imwrite('edges_output.jpg', edges)
    # print(success)

    circles = np.uint16(np.around(circles))
    return [Drum(x, y, radius) for x, y, radius in circles[0]]
