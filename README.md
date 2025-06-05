# 📱 Aruco Pose Estimation

A simple project that overlays 3D shapes (cube, pyramid, and axis) on ArUco markers in real time using a webcam.

---

## 🔍 Overview

This project uses OpenCV and Python to detect ArUco markers and render 3D objects over them in real time.

---

## 📸 Preview

![App Screenshot](https://github.com/udham2511/aruco-pose-estimation/blob/main/demo/output.png)

---

## ✨ Features

- 🎥 Camera calibration and distortion correction  
- 📐 3D position detection of ArUco markers  
- 🟦 Renders 3D cube, pyramid, and axis on detected markers  
- 🎞️ Real-time camera input and rendering  
- 🔄 3D to 2D projection for accurate display  

---

## ⚠️ Important
The script requires a *calibration.npz* file containing your webcam’s camera matrix and distortion coefficients.

To generate your own:
- Capture multiple images of a chessboard pattern using your webcam.
- Use OpenCV’s calibration functions to compute the  camera matrix and distortion coefficients.
- Save them as calibration.npz:

```bash
numpy.savez("calibration.npz", matrix=matrix, distCoeffs=distCoeffs)
```
Place this file in the same directory as the script.

---

## 🛠️ Requirements

- Python 3.x  
- OpenCV (`opencv-contrib-python`)  
- NumPy  

Install requirements:
```bash
pip install opencv-contrib-python numpy
```

## 👨‍💻 Authors

- [@udham2511](https://www.github.com/udham2511)

📬 Connect with me on LinkedIn
- [@udham2511](https://www.linkedin.com/in/udham2511/)