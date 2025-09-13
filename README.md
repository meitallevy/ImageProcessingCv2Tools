# Imgage Processing Tools 

## Background

During high school, I was the first in my robotics team to successfully implement a real-time vision system for object detection and tracking. This system became a critical part of our competition robot, allowing precise detection of circular and colored targets.  

My work was recognized both by my school and the local municipality, where I received an honor for my contributions to technological innovation in youth robotics. This project taught me practical computer vision, problem-solving under pressure, and how persistence leads to real results.

---

## Overview

This repository contains several Python scripts for **computer vision tasks in robotics**:

- Real-time **color detection** using HSV filtering  
- **Contour analysis** and **visualization**  
- **Angle and distance estimation** from camera to target  
- Modular design for testing and competition purposes  
- GUI to visualize results and adjust parameters in real-time  

Among all scripts, **`GenericVision.py`** is the main module: a robust, fully-featured vision system that can detect multiple targets, compute angles and display annotated outputs in real-time.

---

## Files

| File | Description                                                                             |
|------|-----------------------------------------------------------------------------------------|
| `GenericVision.py` | Main vision system. Supports multiple targets, angle estimation, and real-time display. |
| `circular_objects_recognition.py` | Detects circular objects in video streams, with contour visualization.                  |
| `contour_simple_recognition.py` | Simplified contour-based detection for testing or small targets.                        |
| `recognition+distance_for_comp.py` | Competition-ready script: detects targets, computes distance, displays annotations.     |
| `consts.py` | Stores constants like FOV, target size, HSV ranges, and camera parameters.              |
| `hsvimage.png` | An image used for Explaining HSV in dynamic trackbar window                             |
| `README.md` | This file.                                                                              |

---

## Installation

Install dependencies:

```bash
pip install opencv-python numpy imutils
````

---

## Usage

Run the main vision module:

```bash
python GenericVision.py
```

Controls:

* `Space`: Pause / resume
* `Esc` or `q`: Quit


Other scripts can be used for testing or simpler vision tasks:

```bash
python circular_objects_recognition.py
python contour_simple_recognition.py
python recognition+distance_for_comp.py
```
Adjust HSV values and camera exposure in `consts.py` for your specific environment.

---

## Notes

* Designed for high school robotics competitions
* Works best under consistent lighting conditions
* Scripts are modular: you can combine functions to create a custom vision pipeline

---

## Crazy Eyes: Face and Eye Overlay

This repository also includes a **fun real-time face and eye manipulation script**, designed as an experimental exercise in computer vision:

- **Detects faces and eyes** using Haar cascades (`haarcascade_frontalface_default.xml` and `haarcascade_eye.xml`)  
- **Switches eyes** between detected faces in real-time  
- **Optional framing** around faces and eyes to visualize detection  
- Provides a **live GUI** with instructions displayed at the bottom of the window  

**Controls while running:**

- `Space`: Show / hide switched eyes  
- `Esc`: Exit the program  
- `1`: Toggle framing around faces and eyes  


It complements the robotics vision scripts by showing how object detection and manipulation can be applied to different types of targets (faces/eyes) beyond competition objects.
