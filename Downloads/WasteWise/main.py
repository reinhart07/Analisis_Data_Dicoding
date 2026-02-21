from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Suppress TensorFlow warnings (sama seperti classify_waste.py kamu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="WasteWise Classification API",
    description="API deteksi sampah organik dan anorganik untuk aplikasi WasteWise",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Load model (sama persis dengan classify_waste.py kamu)
# ─────────────────────────────────────────────
MODEL_PATH = "./waste_savedmodel"

print("⏳ Loading WasteWise model...")
try:
    model = tf.saved_model.load(MODEL_PATH)
    infer = model.signatures['serving_default']
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None
    infer = None

# Sama persis dengan classify_waste.py kamu
LABELS = ['anorganik', 'organik']

TIPS = {
    'organik': 'Organic waste can be composted! Put it in your compost bin or use it for fertilizer.',
    'anorganik': 'Inorganic waste should be recycled. Take it to the nearest waste bank to earn points!'
}

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "running",
        "app": "WasteWise Classification API",
        "model_loaded": model is not None
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if infer is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg, png, webp)")

    try:
        # Baca dan preprocess gambar (sama dengan classify_waste.py)
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        input_tensor = tf.constant(img_array, dtype=tf.float32)

        # Inferensi
        predictions = infer(input_tensor)
        output_key = list(predictions.keys())[0]
        pred_output = predictions[output_key].numpy()

        # Ambil hasil (sama dengan classify_waste.py)
        predicted_class = int(np.argmax(pred_output[0]))
        confidence = float(pred_output[0][predicted_class]) * 100
        label = LABELS[predicted_class]

        return {
            "label":      label,
            "confidence": f"{confidence:.1f}",  # format sama: "94.3"
            "tips":       TIPS[label],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")