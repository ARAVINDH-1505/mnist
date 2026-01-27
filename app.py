import io
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# ----------------------------
# Model definition (same as training)
# ----------------------------
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


# ----------------------------
# Load model
# ----------------------------
model = Net()
model.load_state_dict(torch.load("mnist_cnn_model.pth", map_location="cpu"))
model.eval()

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="MNIST Digit Recognizer")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    with open("static/index.html") as f:
        return f.read()


class PredictionResponse(BaseModel):
    predicted_digit: int


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28))
    image = np.array(image, dtype=np.float32) / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)
    return image


@app.post("/predict", response_model=PredictionResponse)
async def predict_digit(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)

    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return {"predicted_digit": prediction}


@app.get("/")
def root():
    return {"message": "MNIST FastAPI is running"}
