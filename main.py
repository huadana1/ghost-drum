
import cv2
import mediapipe as mp
import numpy as np
from typing import List
from drum import Drum
from scipy.signal import argrelmax
from pydub import AudioSegment
from pydub.playback import play, _play_with_simpleaudio
import time

CIRCLE_EDGE_DETECTION_THRESHOLD = 60 # higher value finds cleaner edges
CIRCLE_DETECTION_THRESHOLD = 60 # increase reduces false positives
BLUR_FACTOR = 0.3
VALID_MIN_THRESHOLD = 0.02
SMOOTHING_WINDOW = 3
MIN_DETECTION_WINDOW = 15

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

def play_and_record(sound, current_time_ms):
    global audio_output_timeline
    audio_output_timeline = audio_output_timeline.overlay(sound, position=current_time_ms)
    _play_with_simpleaudio(sound)

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


    circles = np.uint16(np.around(circles))

    # TODO: currently looping through the sounds, but we should use set_sound for custom user input sounds at some point once we have an interface? -dana
    return [Drum(x, y, radius, SOUNDS[idx%len(SOUNDS)]) for idx, (x, y, radius) in enumerate(circles[0])]

def smooth_with_window(arr, window_size):
    """Smooth an array using a simple moving average."""
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def detect_hit(hand_landmarks, frame, prev_tip_z):

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils    
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    h, w, _ = frame.shape

    tip = hand_landmarks.landmark[8]
    dip = hand_landmarks.landmark[7]

    tip_z = tip.z
    dip_z = dip.z
    
    if tip_z > dip_z and (prev_tip_z is None or tip_z - prev_tip_z >= 0):
        tip_x_px = int(tip.x * w)
        tip_y_px = int(tip.y * h)
        cv2.circle(frame, (tip_x_px, tip_y_px), 10, (0, 0, 255), -1)
        return tip_x_px, tip_y_px, tip_z
    return None, None, tip_z

def main(vid_path: str) -> list[tuple[int, int]]:
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
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(vid_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(f'Output/{vid_path[:-4]}_video_output.mp4', fourcc, fps, (width, height))
    hits = []
    prev_tip_z_left = None
    prev_tip_z_right = None
    seen_local_min_indices = set()


    # cv2.imwrite(vid_path[:-4] + "_frame_1.jpg", frame) 
    drums = init_drums(vid_path[:-4] + "_frame_1.jpg")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame")
            break

        # read drums from first frame of image
        # hardcode drums for now because not perfect circles in videos
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            x1, y1, z1 = detect_hit(results.multi_hand_landmarks[0], frame, prev_tip_z_left)
            if x1 != None:
                hits.append((x1, y1, z1))
            prev_tip_z_left = z1

            if len(results.multi_hand_landmarks) >= 2:
                x2, y2, z2 = detect_hit(results.multi_hand_landmarks[1], frame, prev_tip_z_right)
                if x2 != None:
                    hits.append((x2, y2, z2))
                prev_tip_z_right = z2
            

        # drawing the hit coords
        if hits:
            z_values = np.array([hit[2] for hit in hits])
            z_values = smooth_with_window(z_values, SMOOTHING_WINDOW)
            # take max bc want to get the farthest point from camera (closer to camera is more neg)
            local_maxima_indices = argrelmax(z_values, order=MIN_DETECTION_WINDOW)[0]
            # print("local min", local_minima_indices)
            for index in local_maxima_indices:
                x, y, z = hits[index]
                # play sound for correct drum only if we have not already done so
                #  Not optimized :(
                if (index not in seen_local_min_indices):
                    for drum in drums:
                        # only allow one sound per hit
                        if drum.hit_in_drum(x,y): 
                            elapsed = int((frame_idx / fps) * 1000)   
                            play_and_record(drum.get_drum_sound(), elapsed)
                            cv2.circle(frame, (x, y), 100, (0, 255, 0), -1)   
                            break
                    seen_local_min_indices.add(index)
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
    audio_output_timeline.export(f'Output/{vid_path[:-4]}_audio_output.wav', format="wav")

    return hits

