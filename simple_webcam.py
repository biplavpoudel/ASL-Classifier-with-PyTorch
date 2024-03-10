from PIL import Image
from CNN_Model import ASLClassifier
import torch
import torchvision.transforms as transforms
import cv2 as cv
import numpy as np

# Load model
model_path = r'D:\ASL Classifier\model\final_model.pt'
final_model = ASLClassifier().to("cuda")
final_model.load_state_dict(torch.load(model_path))
final_model.eval()  # Set to evaluation mode

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_frame(frame):
    image = Image.fromarray(frame)
    image = transform(image).unsqueeze(0)
    return image

def webcam_frame_capture():

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
            image = preprocess_frame(frame)
            # Get model predictions
            with torch.no_grad():
                outputs = final_model(image.to("cuda"))
                print(outputs[0])

            _, predicted = torch.max(outputs.data, 1)
            print(f"Predicted class index: {predicted.item()}")
            idx_to_class = chr(predicted.item()+65)
            print(f"Predicted class:{idx_to_class}")

            cv.putText(frame, idx_to_class, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display frame with hand landmarks
            cv.imshow('Hand Landmarks', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    webcam_frame_capture()
