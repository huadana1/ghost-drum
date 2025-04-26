# Results object contains masks
from ultralytics import YOLO
import cv2
import numpy as np

# Load a segmentation model (like yolov8s-seg)
model = YOLO('yolov8s-seg.pt')  # or your custom trained seg model

# Inference on an image
results = model('Input/gesture_5_frame_1.jpg', 0.2)  # replace with your image


for result in results:
    masks = result.masks.xy  # list of numpy arrays (one per detected object)

    for i, mask in enumerate(masks):
        # mask is a (N, 2) array: [x, y] points outlining the mask
        original_image = result.orig_img.copy()
        # Convert the mask to int and reshape for OpenCV
        contour = np.array(mask, dtype=np.int32)
        contour = contour.reshape((-1, 1, 2))

        # Draw the contour on a blank image (for visualization)
        cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)  # green outline

    # Show the outline
cv2.imshow('ok', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
