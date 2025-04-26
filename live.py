
import cv2
import mediapipe as mp
import numpy as np
from typing import List
from drum import Drum
from scipy.signal import argrelmax
from pydub import AudioSegment
from pydub.playback import play, _play_with_simpleaudio
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

CIRCLE_EDGE_DETECTION_THRESHOLD = 60 # higher value finds cleaner edges
CIRCLE_DETECTION_THRESHOLD = 60 # increase reduces false positives
BLUR_FACTOR = 0.3
VALID_MIN_THRESHOLD = 0.02
SMOOTHING_WINDOW = 3
MAX_DETECTION_WINDOW = 15

CLAP_SOUND = AudioSegment.from_file("Sounds/clap_B_minor.wav")
TOM_SOUND = AudioSegment.from_file("Sounds/big-tom_B_major.wav")
SNARE_SOUND = AudioSegment.from_file("Sounds/clean-snare_C_minor.wav")
CRASH_SOUND = AudioSegment.from_file("Sounds/crash_F_minor.wav")
HI_HAT_SOUND = AudioSegment.from_file("Sounds/hi-hat_B_minor.wav")

SOUNDS = [CRASH_SOUND, SNARE_SOUND, HI_HAT_SOUND, CLAP_SOUND, TOM_SOUND]
MIN_SOUND_DURATION_MS = 1000

# Timeline to track full audio output (e.g., 10 seconds)
# Using to produce the audio soundtrack
audio_timeline_duration = 9000  # milliseconds
audio_output_timeline = AudioSegment.silent(duration=audio_timeline_duration)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

def play_and_record(sound, current_time_ms):
    global audio_output_timeline
    audio_output_timeline = audio_output_timeline.overlay(sound, position=current_time_ms)
    _play_with_simpleaudio(sound)

def init_drums(frame: np.ndarray) -> List[Drum]:
    """
    Initializes Drum objects based on detected circles in an image.

    Params: 
        frame (np.ndarray): A frame/image captured from the video.

    Returns:
        drums (List[Drum]): A list of Drum objects, each with attributes 
        (x, y, radius) corresponding to a detected circle. 
        The 'sound' attribute is initialized to the built-in sounds in the order clap, tom, snare, crash, hi-hat.
    """
    frame_path = 'Input/first_frame.jpg'
    cv2.imwrite(frame_path, frame)
    img = cv2.imread(frame_path)

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


    circles = np.uint16(np.around(circles))

    # TODO: currently looping through the sounds, but we should use set_sound for custom user input sounds at some point once we have an interface? -dana
    return [Drum(x, y, radius, SOUNDS[idx%len(SOUNDS)]) for idx, (x, y, radius) in enumerate(circles[0])]

def draw_drums(drums, frame):
    for drum in drums:
            cv2.circle(frame, (drum.x, drum.y), drum.radius, (255, 0, 0), 2)  # blue circles

def smooth_with_window(arr, window_size):
    """Smooth an array using a simple moving average."""
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def process_drum_hit(drums, x1, y1, frame_idx, fps, frame):
    for drum in drums:
        # only allow one sound per hit
        if drum.hit_in_drum(x1,y1): 
            elapsed = int((frame_idx / fps) * 1000)   
            play_and_record(drum.get_drum_sound(), elapsed)
            cv2.circle(frame, (x1, y1), 100, (0, 255, 0), -1)   
            break

def detect_hit(hand_landmarks, frame, world_z, max_z):    
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    h, w, _ = frame.shape

    tip = hand_landmarks.landmark[8]
    dip = hand_landmarks.landmark[7]

    tip_z = tip.z
    dip_z = dip.z
    print('candidate', world_z)
    print('diff', abs(world_z - max_z))
    if tip_z > dip_z and abs(world_z - max_z) <= 0.03:
        tip_x_px = int(tip.x * w)
        tip_y_px = int(tip.y * h)
        print('Hit detected 💥')
        cv2.circle(frame, (tip_x_px, tip_y_px), 10, (0, 0, 255), -1)
        return True, (tip_x_px, tip_y_px)
    return False, (None, None)

def live() -> list[tuple[int, int]]:
    """
    Determines if a hit was made with the index finger and returns all the coordinates
    of where finger taps (hits) occured.

    Params: 
        vid_path (str): The file path to the video (e.g., 'vid.mp4').

    Returns:
        hits (list[tuple[int, int]]): A list of tuples where each one represents the
        (x, y) coordinate of where a hit occured.
    """

    cap = cv2.VideoCapture(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(f'Output/test.mp4', fourcc, fps, (width, height))

    hits = []

    frame_idx = 0
    drums = []
    max_z = float('-inf')

    print('Software started ✅\n')

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read the first frame")
            break
    
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # read drums from first frame of image
        if frame_idx == 0:
            drums = init_drums(frame)
            print(f'Initialized {len(drums)} 🥁\n')

        draw_drums(drums, frame)

        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:

            # First, update max_z across all hands
            for world_landmarks in results.multi_hand_world_landmarks:
                fingertip_z = world_landmarks.landmark[8].z
                if fingertip_z > max_z:
                    max_z = fingertip_z
                    print('max_z', max_z)

            # Then, process each detected hand
            for hand_landmarks, world_landmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):
                fingertip_z = world_landmarks.landmark[8].z
                hit, (x, y) = detect_hit(hand_landmarks, frame, fingertip_z, max_z)
                if hit: process_drum_hit(drums, x, y, frame_idx, fps, frame)

            
        frame_idx += 1

        cv2.imshow("GESTURE RECOGNITION", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # run the following line in terminal to combine audio and vid
    # ffmpeg -i gesture_6_video_output.mp4 -i gesture_6_audio_output.wav -c:v copy -map 0:v:0 -map 1:a:0 -shortest gesture_6_combined_output.mp4
    # audio_output_timeline.export(f'Output/{vid_path[:-4]}_audio_output.wav', format="wav")

    return hits

