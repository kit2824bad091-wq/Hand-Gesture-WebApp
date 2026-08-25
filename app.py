from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
import math
import threading

app = Flask(__name__)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

mediapipe_lock = threading.Lock()


def calculate_angle(a, b, c):

    ab = (
        a[0] - b[0],
        a[1] - b[1]
    )

    cb = (
        c[0] - b[0],
        c[1] - b[1]
    )

    dot = (
        ab[0] * cb[0]
        +
        ab[1] * cb[1]
    )

    mag_ab = math.sqrt(
        ab[0] ** 2
        +
        ab[1] ** 2
    )

    mag_cb = math.sqrt(
        cb[0] ** 2
        +
        cb[1] ** 2
    )

    if mag_ab == 0 or mag_cb == 0:
        return 0

    value = dot / (mag_ab * mag_cb)

    value = max(
        -1.0,
        min(1.0, value)
    )

    angle = math.degrees(
        math.acos(value)
    )

    return angle


def distance(a, b):

    return math.sqrt(
        (a[0] - b[0]) ** 2
        +
        (a[1] - b[1]) ** 2
    )


def recognize_gesture(frame):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    with mediapipe_lock:

        results = hands.process(
            rgb_frame
        )

    if not results.multi_hand_landmarks:

        return {
            "gesture": "No Hand",
            "fingers": 0
        }

    hand_landmarks = (
        results.multi_hand_landmarks[0]
    )

    h, w, _ = frame.shape

    points = []

    for landmark in hand_landmarks.landmark:

        x = int(
            landmark.x * w
        )

        y = int(
            landmark.y * h
        )

        points.append(
            (x, y)
        )

    # --------------------------------
    # Finger angles
    # --------------------------------

    index_angle = calculate_angle(
        points[5],
        points[6],
        points[8]
    )

    middle_angle = calculate_angle(
        points[9],
        points[10],
        points[12]
    )

    ring_angle = calculate_angle(
        points[13],
        points[14],
        points[16]
    )

    pinky_angle = calculate_angle(
        points[17],
        points[18],
        points[20]
    )

    thumb_angle = calculate_angle(
        points[2],
        points[3],
        points[4]
    )

    # --------------------------------
    # Extended fingers
    # --------------------------------

    index_extended = (
        index_angle > 150
    )

    middle_extended = (
        middle_angle > 150
    )

    ring_extended = (
        ring_angle > 150
    )

    pinky_extended = (
        pinky_angle > 150
    )

    wrist = points[0]

    thumb_tip = points[4]
    thumb_ip = points[3]
    thumb_mcp = points[2]

    thumb_tip_distance = distance(
        wrist,
        thumb_tip
    )

    thumb_ip_distance = distance(
        wrist,
        thumb_ip
    )

    thumb_extended = (
        thumb_angle > 145
        and
        thumb_tip_distance
        >
        thumb_ip_distance * 1.08
    )

    thumb_up = (
        thumb_extended
        and
        thumb_tip[1]
        <
        thumb_mcp[1]
    )

    finger_count = (
        int(thumb_extended)
        +
        int(index_extended)
        +
        int(middle_extended)
        +
        int(ring_extended)
        +
        int(pinky_extended)
    )

    gesture = "Unknown"

    # --------------------------------
    # Thumbs Up
    # --------------------------------

    if (
        thumb_up
        and
        not index_extended
        and
        not middle_extended
        and
        not ring_extended
        and
        not pinky_extended
    ):

        gesture = "Thumbs Up"

    # --------------------------------
    # Victory
    # --------------------------------

    elif (
        index_extended
        and
        middle_extended
        and
        not ring_extended
        and
        not pinky_extended
    ):

        gesture = "Victory"

    # --------------------------------
    # Pointing
    # --------------------------------

    elif (
        index_extended
        and
        not middle_extended
        and
        not ring_extended
        and
        not pinky_extended
    ):

        gesture = "Pointing"

    # --------------------------------
    # Open Palm
    # --------------------------------

    elif (
        index_extended
        and
        middle_extended
        and
        ring_extended
        and
        pinky_extended
    ):

        gesture = "Open Palm"

    # --------------------------------
    # Fist
    # --------------------------------

    elif (
        not thumb_extended
        and
        not index_extended
        and
        not middle_extended
        and
        not ring_extended
        and
        not pinky_extended
    ):

        gesture = "Fist"

    return {
        "gesture": gesture,
        "fingers": finger_count
    }


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "gesture": "No Data",
                "fingers": 0
            })

        image_data = data.get(
            "image"
        )

        if not image_data:

            return jsonify({
                "gesture": "No Image",
                "fingers": 0
            })

        if "," in image_data:

            image_data = (
                image_data.split(
                    ",",
                    1
                )[1]
            )

        image_bytes = (
            base64.b64decode(
                image_data
            )
        )

        np_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:

            return jsonify({
                "gesture": "Invalid Image",
                "fingers": 0
            })

        result = recognize_gesture(
            frame
        )

        print(
            "Detected:",
            result["gesture"],
            "| Fingers:",
            result["fingers"]
        )

        return jsonify(
            result
        )

    except Exception as error:

        print(
            "Prediction Error:",
            error
        )

        return jsonify({
            "gesture": "Error",
            "fingers": 0
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        use_reloader=False
    )