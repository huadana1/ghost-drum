import cv2
import os
from main import init_drums, detect_hit

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

def test_detect_hit():
    vid_paths = [os.path.join('Input', filename) for filename in os.listdir('Input') if filename.endswith(('.mp4'))]

    for vid_path in vid_paths:
        # img = cv2.imread(img_path)
        hits = detect_hit(vid_path)
    
        if hits:
            print("Taps detected at:", hits)
        else:
            print("No taps detected.")

if __name__ == '__main__':
    # test_init_drums()
    test_detect_hit()
    