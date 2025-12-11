from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import shutil
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from collections import Counter

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define router
router = APIRouter(tags=["Detection"])

# Folder structure
UPLOAD_DIR = "uploads"
os.makedirs(f"{UPLOAD_DIR}/videos", exist_ok=True)
os.makedirs(f"{UPLOAD_DIR}/images", exist_ok=True)


# Schemas
class AutismPredictionRequest(BaseModel):
    A1: int
    A2: int
    A3: int
    A4: int
    A5: int
    A6: int
    A7: int
    A8: int
    A9: int
    A10: int
    Age_Mons: int
    Sex: int
    Ethnicity: int
    Jaundice: int
    Family_mem_with_ASD: int


class AutismPredictionResponse(BaseModel):
    prediction: int
    confidence: float


# PyTorch Model
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Adjusted for 224x224 input
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # (224 → 112)
        x = self.pool(self.relu(self.conv2(x)))  # (112 → 56)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# PyTorch Model Setup
PYTORCH_MODEL_PATH = "best_model.pth"  # Root folder
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["autistic", "non-autistic"]
IMAGE_SIZE = (224, 224)
NUM_FRAMES = 10

pytorch_model = CustomCNN(num_classes=NUM_CLASSES).to(DEVICE)
if os.path.exists(PYTORCH_MODEL_PATH):
    pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=DEVICE))
    pytorch_model.eval()
    print(f"✅ PyTorch Model loaded from {PYTORCH_MODEL_PATH}")
else:
    raise FileNotFoundError(
        f"❌ PyTorch Model file {PYTORCH_MODEL_PATH} not found. Please place it in the root folder."
    )

# PyTorch Image Transform
pytorch_transform = transforms.Compose(
    [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]
)


# Helper Functions
def predict_single_image_pytorch(image_path: str) -> dict:
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = pytorch_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = pytorch_model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            _, predicted_idx = torch.max(output, 1)
            prediction = predicted_idx.item()  # 0 or 1
            confidence = probabilities[predicted_idx.item()].item()
        return {"prediction": prediction, "confidence": confidence}
    except Exception as e:
        return {"error": str(e)}


def preprocess_video_pytorch(video_path: str) -> torch.Tensor:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"❌ Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)  # Convert to PIL Image
                frame = pytorch_transform(frame)  # Apply transform
                frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"❌ No valid frames extracted from video: {video_path}")
        return torch.stack(frames).to(
            DEVICE
        )  # Stack into a tensor [NUM_FRAMES, C, H, W]
    except Exception as e:
        print(f"❌ Error preprocessing video: {e}")
        return None


def predict_video_pytorch(video_path: str) -> dict:
    try:
        frames = preprocess_video_pytorch(video_path)
        if frames is None:
            return {"error": "Failed to preprocess video"}
        with torch.no_grad():
            outputs = pytorch_model(frames)  # [NUM_FRAMES, NUM_CLASSES]
            probabilities = torch.nn.functional.softmax(
                outputs, dim=1
            )  # [NUM_FRAMES, NUM_CLASSES]
            _, predicted_indices = torch.max(outputs, 1)  # [NUM_FRAMES]
            binary_predictions = predicted_indices.cpu().numpy()
            counts = Counter(binary_predictions)
            majority_prediction = counts.most_common(1)[0][0]
            confidence = counts[majority_prediction] / len(binary_predictions)
        return {"prediction": int(majority_prediction), "confidence": float(confidence)}
    except Exception as e:
        print(f"❌ Error predicting video: {e}")
        return {"error": str(e)}


# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Autism Detection API"}


@router.post("/predict", response_model=AutismPredictionResponse)
async def predict_autism(request: AutismPredictionRequest):
    try:
        # Sum the A1-A10 scores and apply a simple threshold
        data = [
            request.A1,
            request.A2,
            request.A3,
            request.A4,
            request.A5,
            request.A6,
            request.A7,
            request.A8,
            request.A9,
            request.A10,
        ]
        total_score = sum(data)
        prediction = int(total_score >= 5)  # Threshold: 5 out of 10
        confidence = total_score / 10.0  # Simple confidence based on score
        return {"prediction": prediction, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-images", response_model=List[AutismPredictionResponse])
async def predict_from_images(images: List[UploadFile] = File(...)):
    try:
        if not images:
            raise HTTPException(
                status_code=400, detail="No images uploaded for prediction."
            )
        predictions: List[AutismPredictionResponse] = []
        for image in images:
            image_path = os.path.join(UPLOAD_DIR, "images", image.filename)
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            result = predict_single_image_pytorch(image_path)
            os.remove(image_path)
            if "error" in result:
                print(
                    f"Warning: Skipping image {image.filename} due to error: {result['error']}"
                )
                continue
            predictions.append(
                AutismPredictionResponse(
                    prediction=result["prediction"], confidence=result["confidence"]
                )
            )
        if not predictions:
            raise HTTPException(status_code=500, detail="Failed to predict any images.")
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-videos", response_model=List[AutismPredictionResponse])
async def predict_from_videos(videos: List[UploadFile] = File(...)):
    try:
        if not videos:
            raise HTTPException(status_code=400, detail="No videos uploaded.")

        predictions = []

        for video in videos:
            video_path = os.path.join(UPLOAD_DIR, "videos", video.filename)
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)

            result = predict_video_pytorch(video_path)
            os.remove(video_path)

            if "error" in result:
                print(
                    f"Warning: Skipping video {video.filename} due to error: {result['error']}"
                )
                continue

            predictions.append(
                AutismPredictionResponse(
                    prediction=result["prediction"], confidence=result["confidence"]
                )
            )

        if not predictions:
            raise HTTPException(status_code=500, detail="Failed to predict any videos.")

        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/predict-video", response_model=AutismPredictionResponse)
# async def predict_from_video(video: UploadFile = File(...)):
#     try:
#         video_path = os.path.join(UPLOAD_DIR, "videos", video.filename)
#         with open(video_path, "wb") as buffer:
#             shutil.copyfileobj(video.file, buffer)
#         result = predict_video_pytorch(video_path)
#         os.remove(video_path)
#         if "error" in result:
#             raise HTTPException(status_code=500, detail=result["error"])
#         return AutismPredictionResponse(
#             prediction=result["prediction"], confidence=result["confidence"]
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# Include router in app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
