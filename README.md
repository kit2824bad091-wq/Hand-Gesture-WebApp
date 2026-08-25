# Hand Gesture Recognition Web Application

## Project Overview

This project is a real-time Hand Gesture Recognition Web Application developed using Python, OpenCV, MediaPipe, Flask, HTML, and CSS.

The application captures live video from the webcam, detects hand landmarks, and recognizes different hand gestures in real time.

## Supported Gestures

- 🖐️ Open Palm
- ✊ Fist
- ✌️ Victory
- ☝️ Pointing
- 👍 Thumbs Up

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Flask
- HTML
- CSS

## Features

- Real-time webcam video streaming
- Hand detection using MediaPipe
- 21 hand landmark detection
- Finger counting
- Real-time gesture recognition
- Web-based user interface
- Live gesture name display

## Project Structure

```text
Hand_Gesture_WebApp/
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── gesture_recognition.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Run the Application

Run:

```bash
python app.py
```

Then open the local Flask address shown in the terminal in your web browser.

## How It Works

1. OpenCV captures video from the webcam.
2. MediaPipe detects the hand and its landmarks.
3. The system analyzes finger positions and joint angles.
4. Gesture recognition logic identifies the gesture.
5. Flask streams the processed video to the web interface.
6. The detected gesture and finger count are displayed in real time.

## Application

This project demonstrates how Computer Vision can be used for Human-Computer Interaction through real-time hand gesture recognition.