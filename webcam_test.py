from PIL import Image
from CNN_Model import ASLClassifier
import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open('Input/asl_alphabets/train/C/C45.jpg')
input_image = transform(image).unsqueeze(0)  # Add batch dimension

model_path = r'D:\ASL Classifier\model\final_model.pt'
final_model = ASLClassifier().to("cuda")
final_model.load_state_dict(torch.load(model_path))
final_model.eval()  # Set to evaluation mode

# Get model predictions
with torch.no_grad():
    outputs = final_model(input_image.to("cuda"))

# Example: Get the predicted class index
_, predicted = torch.max(outputs.data, 1)
print(f"Predicted class index: {predicted.item()}")
