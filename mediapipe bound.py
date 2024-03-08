import cv2 as cv
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error opening video stream from webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error. Cannot capture frame. Exiting...")
        break

    # Convert frame to RGB (as Mediapipe requires RGB input)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detect hands
    results = hands.process(frame_rgb)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        # Extract hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on frame
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate bounding box for all landmarks
            x_coords = [frame.shape[1] * landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [frame.shape[0] * landmark.y for landmark in hand_landmarks.landmark]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)

            # Calculate margin for the bounding box
            margin = 30

            # Adjust the bounding box to encompass the entire hand with margin
            x_min = max(0, int(x_min - margin))
            y_min = max(0, int(y_min - margin))
            x_max = min(frame.shape[1], int(x_max + margin))
            y_max = min(frame.shape[0], int(y_max + margin))

            # Ensure the bounding box is a square
            size = max(x_max - x_min, y_max - y_min)
            x_max = min(frame.shape[1], x_min + size)
            y_max = min(frame.shape[0], y_min + size)
            x_min = max(0, x_max - size)
            y_min = max(0, y_max - size)

            # Extract hand region from frame using the adjusted bounding box coordinates
            hand_region = frame[y_min:y_max, x_min:x_max]

            # Display hand region
            cv.imshow('Hand Region', hand_region)

    # Display frame with hand landmarks and bounding box
    cv.imshow('Webcam', frame)

    # Check for exit key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
