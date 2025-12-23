import os
import shutil
import logging
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.preprocessing import image
import gdown

# --------------------------------------------------
# App Initialization
# --------------------------------------------------
app = FastAPI(title="Medical Image Diagnosis API")

templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# Model Download Utility
# --------------------------------------------------
def ensure_models_exist():
    models = {
        "brain_tumor": (
            "braintumor.h5",
            "https://drive.google.com/uc?id=1SVzcoWXlVO-J8z-zusC7C4LouHDIqic1"
        ),
        "tuberculosis": (
            "Tuberculosis_model.h5",
            "https://drive.google.com/uc?id=1t5L-Od5WnETF4VHlWcfY5QjWMtNxZVCW"
        ),
        "pneumonia": (
            "pneumonia_model.h5",
            "https://drive.google.com/uc?id=1KgQyE7-sDnOhMQIlLpjPr63oUfU6W9SX"
        )
    }

    for name, (path, url) in models.items():
        if not os.path.exists(path):
            logging.info(f"Downloading {name} model...")
            gdown.download(url, path, quiet=False)

ensure_models_exist()

# --------------------------------------------------
# Load Models
# --------------------------------------------------
brain_tumor_model = tf.keras.models.load_model("braintumor.h5")
tb_model = tf.keras.models.load_model("Tuberculosis_model.h5")
pneumonia_model = tf.keras.models.load_model("pneumonia_model.h5")

logging.info("All models loaded successfully")

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def save_file(file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

def preprocess_image(
    img_path,
    target_size,
    color_mode="rgb"
):
    img = image.load_img(
        img_path,
        target_size=target_size,
        color_mode=color_mode
    )
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def binary_prediction(model, img, threshold=0.5):
    prediction = model.predict(img)
    confidence = float(prediction[0][0])
    return confidence >= threshold, confidence

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/diagnose/")
async def diagnose_image(
    diagnosis_type: str = Form(...),
    scan: UploadFile = Form(...)
):
    file_path = os.path.join(UPLOAD_DIR, "scan.jpg")
    save_file(scan, file_path)

    try:
        if diagnosis_type == "brain_tumor":
            img = preprocess_image(file_path, (224, 224), "rgb")
            preds = brain_tumor_model.predict(img)
            class_index = np.argmax(preds)

            classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
            return {
                "diagnosis": classes[class_index],
                "confidence": float(np.max(preds)),
                "model": "Brain Tumor CNN"
            }

        elif diagnosis_type == "tuberculosis":
            img = preprocess_image(file_path, (64, 64), "rgb")

            if len(tb_model.input_shape) == 2:
                img = img.reshape(1, -1)

            detected, confidence = binary_prediction(tb_model, img)
            return {
                "diagnosis": "Tuberculosis" if detected else "Normal",
                "confidence": confidence,
                "model": "TB Detection Model"
            }

        elif diagnosis_type == "pneumonia":
            img = preprocess_image(file_path, (150, 150), "grayscale")
            detected, confidence = binary_prediction(pneumonia_model, img)

            return {
                "diagnosis": "Pneumonia" if detected else "Normal",
                "confidence": confidence,
                "model": "Pneumonia CNN"
            }

        else:
            return {"error": "Invalid diagnosis type selected"}

    except Exception as e:
        logging.error(str(e))
        return {"error": "Prediction failed", "details": str(e)}

# --------------------------------------------------
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
