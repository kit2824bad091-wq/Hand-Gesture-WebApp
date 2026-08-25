import cv2
import mediapipe as mp
import math

# --------------------------------------------------
# 1. MediaPipe Setup
# --------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# --------------------------------------------------
# 2. Angle Function
# --------------------------------------------------

def calculate_angle(a, b, c):
    """
    Calculate angle ABC
    b is the middle point
    """

    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])

    dot = ab[0] * cb[0] + ab[1] * cb[1]

    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if mag_ab == 0 or mag_cb == 0:
        return 0

    cos_angle = dot / (mag_ab * mag_cb)

    # Avoid math domain error
    cos_angle = max(-1.0, min(1.0, cos_angle))

    angle = math.degrees(math.acos(cos_angle))

    return angle


# --------------------------------------------------
# 3. Distance Function
# --------------------------------------------------

def distance(a, b):

    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


# --------------------------------------------------
# 4. Start Camera
# --------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected!")
    exit()

print("Gesture Recognition Started!")
print("Press Q to quit.")


# --------------------------------------------------
# 5. Main Loop
# --------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:
        print("Failed to read camera frame!")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    gesture = "No Hand"
    finger_count = 0

    # --------------------------------------------------
    # 6. Hand Detected
    # --------------------------------------------------

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            h, w, _ = frame.shape

            points = []

            # Convert landmark values into pixels
            for lm in hand_landmarks.landmark:

                x = int(lm.x * w)
                y = int(lm.y * h)

                points.append((x, y))

            # --------------------------------------------------
            # 7. Calculate Finger Angles
            # --------------------------------------------------

            # Index finger
            index_angle = calculate_angle(
                points[5],
                points[6],
                points[8]
            )

            # Middle finger
            middle_angle = calculate_angle(
                points[9],
                points[10],
                points[12]
            )

            # Ring finger
            ring_angle = calculate_angle(
                points[13],
                points[14],
                points[16]
            )

            # Pinky finger
            pinky_angle = calculate_angle(
                points[17],
                points[18],
                points[20]
            )

            # Thumb
            thumb_angle = calculate_angle(
                points[2],
                points[3],
                points[4]
            )

            # --------------------------------------------------
            # 8. Check Extended Fingers
            # --------------------------------------------------

            index_extended = index_angle > 150
            middle_extended = middle_angle > 150
            ring_extended = ring_angle > 150
            pinky_extended = pinky_angle > 150

            # --------------------------------------------------
            # 9. Improved Thumb Detection
            # --------------------------------------------------

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

            # Thumb should be straight and away from palm
            thumb_extended = (
                thumb_angle > 145
                and thumb_tip_distance >
                thumb_ip_distance * 1.08
            )

            # Thumb pointing upward
            thumb_up = (
                thumb_extended
                and thumb_tip[1] < thumb_mcp[1]
            )

            # --------------------------------------------------
            # 10. Count Fingers
            # --------------------------------------------------

            finger_count = (
                int(thumb_extended)
                + int(index_extended)
                + int(middle_extended)
                + int(ring_extended)
                + int(pinky_extended)
            )

            # --------------------------------------------------
            # 11. Gesture Recognition
            # --------------------------------------------------

            # 👍 THUMBS UP
            if (
                thumb_up
                and not index_extended
                and not middle_extended
                and not ring_extended
                and not pinky_extended
            ):

                gesture = "Thumbs Up"

            # ✌ VICTORY
            elif (
                index_extended
                and middle_extended
                and not ring_extended
                and not pinky_extended
            ):

                gesture = "Victory"

            # ☝ POINTING
            elif (
                index_extended
                and not middle_extended
                and not ring_extended
                and not pinky_extended
            ):

                gesture = "Pointing"

            # 🖐 OPEN PALM
            elif (
                index_extended
                and middle_extended
                and ring_extended
                and pinky_extended
            ):

                gesture = "Open Palm"

            # ✊ FIST
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

            # --------------------------------------------------
            # 12. Draw Hand Landmarks
            # --------------------------------------------------

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # --------------------------------------------------
            # 13. Debug Values
            # --------------------------------------------------

            cv2.putText(
                frame,
                "Thumb: " + str(thumb_extended),
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Thumb Angle: " + str(int(thumb_angle)),
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # --------------------------------------------------
    # 14. Display Gesture
    # --------------------------------------------------

    cv2.putText(
        frame,
        "Gesture: " + gesture,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------
    # 15. Finger Count
    # --------------------------------------------------

    cv2.putText(
        frame,
        "Fingers: " + str(finger_count),
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # --------------------------------------------------
    # 16. Display Camera
    # --------------------------------------------------

    cv2.imshow(
        "Hand Gesture Recognition",
        frame
    )

    # Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# 17. Close
# --------------------------------------------------

cap.release()
hands.close()
cv2.destroyAllWindows()