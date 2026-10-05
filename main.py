
import cv2
import mediapipe as mp

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not access the webcam.")
    print("Close other apps using the camera and try again.")
    raise SystemExit(1)

print("GestureOS hand tracking started!")
print("Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read a webcam frame.")
        break

    # Mirror the camera view
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(rgb_frame)

    # Draw the 21 hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        cv2.putText(
            frame,
            "HAND DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 180),
            2
        )
    else:
        cv2.putText(
            frame,
            "SHOW YOUR HAND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 200, 255),
            2
        )

    cv2.imshow("GestureOS | Hand Tracking", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()