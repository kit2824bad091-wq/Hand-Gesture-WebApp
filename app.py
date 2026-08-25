from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

camera = cv2.VideoCapture(0)


def calculate_angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])

    dot = ab[0] * cb[0] + ab[1] * cb[1]

    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if mag_ab == 0 or mag_cb == 0:
        return 0

    value = dot / (mag_ab * mag_cb)
    value = max(-1.0, min(1.0, value))

    return math.degrees(math.acos(value))


def distance(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def generate_frames():

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        gesture = "No Hand"
        finger_count = 0

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                h, w, _ = frame.shape

                points = []

                for lm in hand_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)

                    points.append((x, y))

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

                index_extended = index_angle > 150
                middle_extended = middle_angle > 150
                ring_extended = ring_angle > 150
                pinky_extended = pinky_angle > 150

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
                    and thumb_tip_distance >
                    thumb_ip_distance * 1.08
                )

                thumb_up = (
                    thumb_extended
                    and thumb_tip[1] < thumb_mcp[1]
                )

                finger_count = (
                    int(thumb_extended)
                    + int(index_extended)
                    + int(middle_extended)
                    + int(ring_extended)
                    + int(pinky_extended)
                )

                if (
                    thumb_up
                    and not index_extended
                    and not middle_extended
                    and not ring_extended
                    and not pinky_extended
                ):
                    gesture = "Thumbs Up"

                elif (
                    index_extended
                    and middle_extended
                    and not ring_extended
                    and not pinky_extended
                ):
                    gesture = "Victory"

                elif (
                    index_extended
                    and not middle_extended
                    and not ring_extended
                    and not pinky_extended
                ):
                    gesture = "Pointing"

                elif (
                    index_extended
                    and middle_extended
                    and ring_extended
                    and pinky_extended
                ):
                    gesture = "Open Palm"

                elif (
                    not thumb_extended
                    and not index_extended
                    and not middle_extended
                    and not ring_extended
                    and not pinky_extended
                ):
                    gesture = "Fist"

                else:
                    gesture = "Unknown"

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(
            frame,
            "Gesture: " + gesture,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Fingers: " + str(finger_count),
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)