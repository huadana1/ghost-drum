# Ghost Drum 

Ghost Drum is a multimodal interactive system that allows users to define virtual drumheads, associate each with a custom sound via voice and gesture, and trigger playback through hand gestures alone. It integrates computer vision, speech recognition, and gesture tracking to enable hands-free musical interaction using simple visual markers.

---

## Table of Contents

| File/Folder       | Description |
|-------------------|-------------|
| `v2.py`           | Main system entry point. Handles full pipeline: drum detection, voice input, gesture recognition, and sound playback. |
| `v1.py`           | Older version of the pipeline. Not actively used but useful for reference or debugging. |
| `drum.py`         | Defines the `Drum` class used throughout the system to store position and sound metadata. |
| `test.py`         | Contains testing functions for circle detection and system behavior. Can be used to run live demos. |
| `Input/`          | Folder containing input images (e.g., for drum detection). |
| `Output/`         | Generated output videos and annotated images are saved here. |
| `Sounds/`         | Sound files (e.g., `snare.wav`, `clap.wav`, `cat.mp3`) used in both predefined and custom sound libraries. |

---

##  Setup Instructions

### Prerequisites

Ensure your machine has:

- Python 3.11 (not 3.13!)
- A webcam (can be a phone camera connected to laptop)
- Working microphone
- Paper circle cutouts (used as drum heads)
- A tripod to set up the camera

### Steps

1. Install necessary packages using pip install -r requirements.txt.
2. Position the webcam using a tripod so that it clearly captures the paper circles laid out as drumheads.
3. Run `python3 test.py` to start the system. This will
    - Start your webcam
    - Detect circular drum heads and outline them
    - Prompt you to point at eah drum and say a sound name displayed on the interface (e.g., "cat")
4. After associating drums to sounds, you can start playing the drums/instruments by tapping with your index finger on the paper circles!

