
import cv2
import mediapipe as mp
import numpy as np
from typing import List
from drum import Drum
from scipy.signal import argrelmax
from pydub import AudioSegment
from pydub.playback import play, _play_with_simpleaudio
import time
import threading
import sys
import os
from vosk import Model, KaldiRecognizer
import pyaudio
import queue
import json

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

CIRCLE_EDGE_DETECTION_THRESHOLD = 60 # higher value finds cleaner edges
CIRCLE_DETECTION_THRESHOLD = 60 # increase reduces false positives
BLUR_FACTOR = 0.3
VALID_MIN_THRESHOLD = 0.02
SMOOTHING_WINDOW = 3
MAX_DETECTION_WINDOW = 15
HIT_DETECTION_THRESHOLD = 0.03
HIT_WINDOW = 3
DISTANCE_BETWEEN_CIRCLES = 100

CLAP_SOUND = AudioSegment.from_file("Sounds/clap_B_minor.wav")
TOM_SOUND = AudioSegment.from_file("Sounds/big-tom_B_major.wav")
SNARE_SOUND = AudioSegment.from_file("Sounds/clean-snare_C_minor.wav")
CRASH_SOUND = AudioSegment.from_file("Sounds/crash_F_minor.wav")
HI_HAT_SOUND = AudioSegment.from_file("Sounds/hi-hat_B_minor.wav")

SOUND_LIBRARY = {
    "clap": CLAP_SOUND,
    "tom": TOM_SOUND,
    "snare": SNARE_SOUND,
    "crash": CRASH_SOUND,
    "hi hat": HI_HAT_SOUND
}

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
        minDist=DISTANCE_BETWEEN_CIRCLES, 
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

def detect_hit(hand_landmarks, frame, max_hit_diff):    
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    h, w, _ = frame.shape

    tip = hand_landmarks.landmark[8]
    mcp = hand_landmarks.landmark[5]

    tip_z = tip.z
    mcp_z = mcp.z

    diff = tip_z - mcp_z
    # if tip_z > mcp_z and abs(diff - max_hit_diff) <= HIT_DETECTION_THRESHOLD:
    if tip_z > mcp_z:
        tip_x_px = int(tip.x * w)
        tip_y_px = int(tip.y * h)
        print('Hit detected 💥')
        cv2.putText(frame, f'Tip: {tip_z} Mcp: {mcp_z}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
        cv2.circle(frame, (tip_x_px, tip_y_px), 10, (0, 0, 255), -1)
        return True, (tip_x_px, tip_y_px)
    return False, (None, None)

def record_audio(sounds, num_drums):
    # model = Model("vosk-model-small-en-us-0.15")
    model = Model("vosk-model-en-us-0.22-lgraph")

    recognizer = KaldiRecognizer(model, 16000)

    # Start audio stream
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
    stream.start_stream()
    print("Listening...")

    while len(sounds) < num_drums:
        data = stream.read(4000, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())

            for sound_name in SOUND_LIBRARY:
                if sound_name in result['text']:
                    sounds.append(sound_name)        
                    print(f'Sound {sound_name} successfully added!') 
        else:
            partial_result = recognizer.PartialResult()
            # print(partial_result)

    print("Voice recognition ended")

def assign_drum_sound() -> list[tuple[int, int]]:


    cap = cv2.VideoCapture(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(f'Output/test.mp4', fourcc, fps, (width, height))
    frame_idx = 0
    drums_detected = []
    sounds = []

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read the first frame")
            break

         # read drums from first frame of image
        if frame_idx == 0:
            drums = init_drums(frame)
            print(f'Initialized {len(drums)} 🥁\n')
            
            audio_thread = threading.Thread(
                target=lambda: record_audio(sounds, len(drums))
            )
            audio_thread.start()

        draw_drums(drums, frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:

            # Then, process each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                x, y = hand_landmarks.landmark[8].x,  hand_landmarks.landmark[8].y
                for drum in drums:
                    if drum.hit_in_drum(x * width, y * height): 
                        if drum not in drums_detected:
                            drums_detected.append(drum)

        frame_idx += 1

        if not audio_thread.is_alive():
            for drum, sound_name in zip(drums_detected, sounds):
                drum.set_drum_sound(sound_name, SOUND_LIBRARY[sound_name])
                cv2.putText(frame, drum.sound_name, (drum.x, drum.y), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

        cv2.imshow("GESTURE RECOGNITION", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break
        
    cap.release()
    out.release()
    audio_thread.join()

    

    cv2.destroyAllWindows()

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

    last_hit_frame_idx = 0

    frame_idx = 0
    drums = []
    max_hit_diff = float('-inf')

    print('Software started ✅\n')

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read the first frame")
            break

        # First, update max_z across all hands
        if 10 <= frame_idx and frame_idx <= 100:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    tip_z = hand_landmarks.landmark[8].z
                    mcp_z = hand_landmarks.landmark[5].z

                    diff = tip_z - mcp_z
                    if diff > max_hit_diff:
                        max_hit_diff = diff
            
    
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # read drums from first frame of image
        if frame_idx == 0:
            drums = init_drums(frame)
            print(f'Initialized {len(drums)} 🥁\n')

        draw_drums(drums, frame)

    
        if results.multi_hand_landmarks and results.multi_hand_world_landmarks:

            # Then, process each detected hand
            for hand_landmarks in results.multi_hand_landmarks:

                hit, (x, y) = detect_hit(hand_landmarks, frame, max_hit_diff)
                if hit and frame_idx - last_hit_frame_idx >= HIT_WINDOW: 
                    last_hit_frame_idx = frame_idx
                    process_drum_hit(drums, x, y, frame_idx, fps, frame)

            
        if frame_idx < 100:
            cv2.putText(frame, 'Place index finger on table', (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

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


