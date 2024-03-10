from PIL import Image
from CNN_Model import ASLClassifier
import torch
import torchvision.transforms as transforms
import cv2 as cv
import mediapipe as mp
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_frame(frame):
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image = transform(image).unsqueeze(0)
    return image.to("cuda")


def bounding_box(frame):
    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

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
            return hand_region


def webcam_frame_capture():
    # Load model
    model_path = r'D:\ASL Classifier\model\final_model.pt'
    final_model = ASLClassifier().to("cuda")
    final_model.load_state_dict(torch.load(model_path))
    final_model.eval()  # Set to evaluation mode

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream from webcam")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error. Cannot capture frame. Exiting...")
                break

            hand_region = bounding_box(frame)

            if hand_region is not None:  # Check if hand is detected
                image = preprocess_frame(hand_region)
                # Get model predictions
                with torch.no_grad():
                    outputs = final_model(image.to("cuda"))

                _, predicted = torch.max(outputs.data, 1)
                print(f"Predicted class index: {predicted.item()}")
                idx_to_class = chr(predicted.item()+65)
                print(f"Predicted class:{idx_to_class}")

                cv.putText(frame, idx_to_class, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Display hand region
                cv.imshow('Hand Region', hand_region)

            # Display frame with hand landmarks
            cv.imshow('Hand Landmarks', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    webcam_frame_capture()
