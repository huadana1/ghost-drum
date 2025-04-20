
import cv2
import mediapipe as mp
import numpy as np
from typing import List
from drum import Drum
from scipy.signal import argrelmin
from pydub import AudioSegment
from pydub.playback import play

CIRCLE_EDGE_DETECTION_THRESHOLD = 60 # higher value finds cleaner edges
CIRCLE_DETECTION_THRESHOLD = 60 # increase reduces false positives
BLUR_FACTOR = 0.3
VALID_MIN_THRESHOLD = 0.02
SMOOTHING_WINDOW = 3
MIN_DETECTION_WINDOW = 7

CLAP_SOUND = AudioSegment.from_file("Sounds/clap_B_minor.wav")
TOM_SOUND = AudioSegment.from_file("Sounds/big-tom_B_major.wav")
SNARE_SOUND = AudioSegment.from_file("Sounds/clean-snare_C_minor.wav")
CRASH_SOUND = AudioSegment.from_file("Sounds/crash_F_minor.wav")
HI_HAT_SOUND = AudioSegment.from_file("Sounds/hi-hat_B_minor.wav")

SOUNDS = [CRASH_SOUND, SNARE_SOUND, HI_HAT_SOUND, CLAP_SOUND, TOM_SOUND]
MIN_SOUND_DURATION_MS = 1000

def pad_sounds():
    """Modifies SOUNDS in place to last for MIN_SOUND_DURATION_MS"""
    for i, sound in enumerate(SOUNDS):
        if len(sound) < MIN_SOUND_DURATION_MS:
            pad_duration = MIN_SOUND_DURATION_MS - len(sound)
            sound += AudioSegment.silent(duration=pad_duration)
            SOUNDS[i] = sound
pad_sounds()

def init_drums(img_path: str) -> List[Drum]:
    """
    Initializes Drum objects based on detected circles in an image.

    Params: 
        img_path (str): The file path to the image (e.g., 'image.png').

    Returns:
        drums (List[Drum]): A list of Drum objects, each with attributes 
        (x, y, radius) corresponding to a detected circle. 
        The 'sound' attribute is initialized to the built-in sounds in the order clap, tom, snare, crash, hi-hat.
    """
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    max_radius = min(width, height)
    kernel_size = int(min(height, width) * BLUR_FACTOR)
    kernel_size += 1 if kernel_size % 2 == 0 else 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 5)

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

    # TODO: currently looping through the sounds, but we should use set_sound for custom user input sounds at some point once we have an interface? -dana
    return [Drum(x, y, radius, SOUNDS[idx%len(SOUNDS)]) for idx, (x, y, radius) in enumerate(circles[0])]

def smooth_with_window(arr, window_size):
    """Smooth an array using a simple moving average."""
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def helper()
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
    prev_tip_z = None
    seen_local_min_indices = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame")
            break

        # read drums from first frame of image
        cv2.imwrite(vid_path[:-4] + "_frame_1.jpg", frame) 
        # drums = init_drums(vid_path[:-3] + "_frame_1.jpg")
        # hardcode drums for now because not perfect circles in videos
        drums = [Drum(650, 300, 40, CRASH_SOUND), Drum(700, 300, 40, HI_HAT_SOUND), Drum(1450, 400, 200, TOM_SOUND)]

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                tip = hand_landmarks.landmark[8]
                dip = hand_landmarks.landmark[7]

                tip_z = tip.z
                dip_z = dip.z
                
                if tip_z > dip_z and (prev_tip_z is None or tip_z - prev_tip_z > 0.05):
                    if not tapping:
                        tip_x_px = int(tip.x * w)
                        tip_y_px = int(tip.y * h)
                        cv2.circle(frame, (tip_x_px, tip_y_px), 10, (0, 0, 255), -1)
                        hits.append((tip_x_px, tip_y_px, tip.z))
                        tapping = True
                        

                else:
                    tapping = False

                prev_tip_z = tip_z

        # drawing the hit coords
        if hits:
            z_values = np.array([hit[2] for hit in hits])
            z_values = smooth_with_window(z_values, SMOOTHING_WINDOW)
            local_minima_indices = argrelmin(z_values, order=MIN_DETECTION_WINDOW)[0]
            # print("local min", local_minima_indices)
            for index in local_minima_indices:
                x, y, z = hits[index]

                # play sound for correct drum only if we have not already done so
                #  Not optimized :(
                if (index not in seen_local_min_indices):
                    for drum in drums:
                        # only allow one sound per hit
                        if drum.hit_in_drum(x,y):
                            drum.play()
                            break
                    seen_local_min_indices.add(index)

                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)            
                

        cv2.imshow("GESTURE RECOGNITION", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return hits

