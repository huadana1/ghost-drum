
import cv2
import mediapipe as mp
import numpy as np
from typing import List
from drum import Drum

CIRCLE_EDGE_DETECTION_THRESHOLD = 110 # higher value finds cleaner edges
CIRCLE_DETECTION_THRESHOLD = 200 # increase reduces false positives
BLUR_FACTOR = 0.2

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


def detect_hit(vid_path: str) -> list[tuple[int, int]]:
    """
    Determines if a hit was made with the index finger and returns all the coordinates
    of where finger taps (hits) occured.

    Params: 
        vid_path (str): The file path to the video (e.g., 'vid.mp4').

    Returns:
        hits (list[tuple[int, int]]): A list of tuples where each one represents the
        (x, y) coordinate of where a hit occured.
    """
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(vid_path)
    hits = []
    tapping = False
    prev_tip_y = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                tip = hand_landmarks.landmark[8]
                dip = hand_landmarks.landmark[7]

                tip_y = tip.y
                dip_y = dip.y

                if tip_y > dip_y and (prev_tip_y is None or tip_y - prev_tip_y > 0.02):
                    if not tapping:
                        tip_x_px = int(tip.x * w)
                        tip_y_px = int(tip.y * h)
                        hits.append((tip_x_px, tip_y_px))
                        tapping = True

                else:
                    tapping = False

                prev_tip_y = tip_y

        # drawing the hit coords
        for x, y in hits:
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        cv2.imshow("GESTURE RECOGNITION", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return hits

