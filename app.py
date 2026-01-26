%%writefile app.py
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import io

# 1. Define the Neural Network Architecture (must be the same as trained)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Initialize FastAPI app
app = FastAPI()

# 3. Load the trained model
model_path = 'mnist_cnn_model.pth'
model = Net() # Instantiate the model

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set model to evaluation mode
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle error, e.g., exit or raise exception

# 4. Define prediction function
def predict_image(image_bytes):
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('L') # Convert to grayscale
    image = image.resize((28, 28)) # Resize to 28x28
    img_array = np.array(image, dtype=np.float32) / 255.0 # Normalize

    # Convert to tensor and reshape for model input
    image_tensor = torch.tensor(img_array).reshape(1, 1, 28, 28)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()

# 5. Define a Pydantic model for response (optional but good practice)
class PredictionResponse(BaseModel):
    predicted_digit: int

# 6. Define prediction endpoint
@app.post("/predict/", response_model=PredictionResponse)
async def predict_digit(file: UploadFile = File(...)):
    image_bytes = await file.read()
    predicted_digit = predict_image(image_bytes)
    return {"predicted_digit": predicted_digit}

@app.get("/")
async def root():
    return {"message": "Welcome to the MNIST Digit Recognizer API! Send a POST request to /predict/ with an image file."}
