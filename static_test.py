from PIL import Image
from CNN_Model import ASLClassifier
import torch
import torchvision.transforms as transforms

# Define the path to the input image
image_path = 'EXTERNAL_W.jpg'
# image_path = 'Input/asl_alphabets/train/C/C909.jpg'

# Define the transformation to be applied to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the input image
input_image = Image.open(image_path)

# Apply the defined transformation to the input image
input_image = transform(input_image).unsqueeze(0)  # Add batch dimension

# Load the ASL classifier model
model_path = r'D:\ASL Classifier\model\final_model.pt'
final_model = ASLClassifier().to("cuda")
final_model.load_state_dict(torch.load(model_path))
final_model.eval()  # Set to evaluation mode

# Perform prediction
with torch.no_grad():
    outputs = final_model(input_image.to("cuda"))

# Get the predicted class index
_, predicted = torch.max(outputs.data, 1)
print(f"Predicted class index: {predicted.item()}")
idx_to_class = chr(predicted.item() + 65)
print(f"Predicted class: {idx_to_class}")
