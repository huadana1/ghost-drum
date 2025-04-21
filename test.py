import numpy as np
from scipy.signal import argrelmin
import cv2
import os
from main import init_drums, detect_hit, SOUNDS, main
from drum import Drum
from pydub.playback import play

def test_init_drums():
    img_paths = img_paths = [os.path.join('Input', filename) for filename in os.listdir('Input') if filename.endswith(('.png', '.jpg', '.jpeg'))]
    green = (57, 255, 20)
    thickness = 3

    for img_path in img_paths:
        img = cv2.imread(img_path)
        drums = init_drums(img_path)

        for drum in drums:
            cv2.circle(img, (drum.x, drum.y), drum.radius, green, thickness) 
        
        filename = os.path.basename(img_path)
        output_path = 'Output/test_' + filename
        success = cv2.imwrite(output_path, img)
        print(f"Saved to {output_path}: {success}")

def test_main():
    vid_paths = [os.path.join('Input', filename) for filename in os.listdir('Input') if filename.endswith(('.mp4'))]

    # for vid_path in vid_paths:
        # img = cv2.imread(img_path)
    hits = main("Input/gesture_8.mp4")

    if hits:
        print("Taps detected at:", hits)
    else:
        print("No taps detected.")

def test_play_sound():
    # for sound in SOUNDS:
    #     play(sound)

    drums = [Drum(650, 300, 40, SOUNDS[0]), Drum(700, 300, 40, SOUNDS[0]), Drum(1450, 400, 200, SOUNDS[0])]

    hits = [(678, 243, 0.08209886401891708), (689, 251, 0.07918279618024826), (696, 257, 0.09394631534814835), (702, 266, 0.08539596199989319), (709, 271, 0.12486223876476288)]

    for hit in hits:
        x, y, z = hit

        for drum in drums:
            if drum.hit_in_drum(x,y):
                # only allow one sound per hit
                break

if __name__ == '__main__':
    # test_init_drums()
    test_main()
    # test_play_sound()
    