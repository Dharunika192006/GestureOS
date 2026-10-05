
import cv2
import mediapipe as mp
import math

# MediaPipe hand tracking setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not access webcam.")
    raise SystemExit(1)

print("GestureOS started!")
print("Show an open palm to activate the system.")
print("Press Q to quit.")

def is_open_palm(hand):
    points = hand.landmark

    # Check whether the four fingers are extended
    fingers_open = (
        points[8].y < points[6].y and
        points[12].y < points[10].y and
        points[16].y < points[14].y and
        points[20].y < points[18].y
    )

    # Check whether the thumb is spread away from the palm
    thumb_distance = math.hypot(
        points[4].x - points[5].x,
        points[4].y - points[5].y
    )
    thumb_open = thumb_distance > 0.10

    return fingers_open and thumb_open


while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Mirror the camera view
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    status = "SHOW YOUR HAND"
    status_color = (0, 200, 255)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if is_open_palm(hand_landmarks):
                status = "SYSTEM ACTIVATED"
                status_color = (0, 255, 120)
            else:
                status = "HAND DETECTED"
                status_color = (0, 200, 255)

    # Futuristic-style status panel
    cv2.rectangle(frame, (10, 10), (470, 85), (20, 20, 35), -1)
    cv2.rectangle(frame, (10, 10), (470, 85), (255, 180, 0), 2)

    cv2.putText(
        frame, "GESTUREOS",
        (25, 38), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2
    )

    cv2.putText(
        frame, status,
        (25, 68), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, status_color, 2
    )

    cv2.imshow("GestureOS | Open-Palm Activation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()