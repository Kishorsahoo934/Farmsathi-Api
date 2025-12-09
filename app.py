import os
import io
import re
import json
import pickle
import joblib
import uvicorn
import asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import base64
from groq import Groq

# ========== GLOBAL CONFIG ==========
API_KEY = os.getenv("OPENROUTER_API_KEY")
# System-level instruction for the chatbot and recommendation generation.
# This prompt instructs the model to return short, point-wise answers.
SYSTEM_PROMPT = (
    "You are FarmSathi, a friendly agricultural assistant who speaks in simple, easy-to-understand language. "
    "Format your responses like this:\n\n"
        "IMPORTANT: Only provide information about crops that are suitable for and commonly grown in Odisha state. "
        "Focus on local farming practices, climate conditions, and crop varieties specific to Odisha.\n\n"
        "For lists of crops (specific to Odisha):\n"
    "1. First item\n"
    "2. Second item\n"
    "3. Third item\n\n"
    "For explanations and steps:\n"
    "• Use bullet points\n"
    "• Write in simple words that farmers understand easily\n"
    "• Avoid technical terms - explain them if needed\n"
    "• Keep each point short (1-2 simple sentences)\n"
    "• Put each point on a new line\n\n"
    "Reference: Corn Common Rust (use this when the disease is Corn Common Rust)\n"
    "- Cause: Fungus (Puccinia sorghi) that prefers cool, wet, humid conditions.\n"
    "- Symptoms: Small raised orange-brown pustules on top/bottom of leaves; may release rusty spores when rubbed.\n"
    "- Why it matters: Damages leaves, reducing photosynthesis and yield.\n\n"
    "Short Treatment & Prevention (point-wise):\n"
    "1. Choose rust-resistant corn varieties when possible.\n"
    "2. Remove or bury crop debris after harvest; manage weeds to reduce disease reservoirs.\n"
    "3. Improve airflow: avoid dense planting; space rows to reduce humidity.\n"
    "4. If severe, apply an appropriate fungicide early; follow label instructions and safety precautions.\n"
    "5. Consult local extension services for region-specific fungicide recommendations.\n\n"
    "Always keep answers short, actionable, and in point form. If the user asks for more details, offer a short summary and suggest next steps."
)

app = FastAPI(
    title="🌾 FarmSathi Unified Agriculture API",
    description="One API for Crop Recommendation, Fertilizer, Statistics, Chatbot, and Disease Detection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 1️⃣ CROP RECOMMENDATION ENDPOINT
# ======================================================
try:
    with open('crop_recommendation_model.pkl', 'rb') as file:
        crop_model = pickle.load(file)
    print("✅ Crop recommendation model loaded.")
except Exception as e:
    print(f"❌ Error loading crop model: {e}")
    crop_model = None

@app.post("/crop-recommend")
async def crop_recommend(
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...)
):
    if crop_model is None:
        raise HTTPException(status_code=500, detail="Crop model not loaded properly.")

    input_data = [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]]
    prediction = crop_model.predict(input_data)
    return {"recommended_crop": prediction[0]}
# # ======================================================
# # 3️⃣ FERTILIZER RECOMMENDATION ENDPOINT
# ======================================================
FERT_MODEL_PATH = 'fertilizer_recommendation_model_latest.joblib'
FERT_META_PATH = 'fertilizer_model_metadata_latest.json'

try:
    fert_model = joblib.load(FERT_MODEL_PATH)
    with open(FERT_META_PATH, 'r') as f:
        fert_metadata = json.load(f)
    FERT_FEATURES = fert_metadata['feature_info']['feature_columns']
    print("✅ Fertilizer model loaded.")
except Exception as e:
    print(f"❌ Error loading fertilizer model: {e}")
    fert_model, FERT_FEATURES = None, []

FERTILIZER_MAP = {
    0: '14-35-14', 1: '28-28', 2: 'DAP', 3: 'MOP', 4: 'Potash', 5: 'SSP', 6: 'Urea'
}
SOIL_TYPES = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
CROP_TYPES = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 
              'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

@app.post("/fertilizer-recommend")
async def fertilizer_recommend(
    temp: float = Form(...),
    humidity: float = Form(...),
    moisture: float = Form(...),
    nitrogen: float = Form(...),
    phosphorous: float = Form(...),
    potassium: float = Form(...),
    ph: float = Form(...),
    soil_type: str = Form(...),
    crop_type: str = Form(...)
):
    if fert_model is None:
        raise HTTPException(status_code=500, detail="Fertilizer model not available.")

    soil_encoded = SOIL_TYPES.index(soil_type)
    crop_encoded = CROP_TYPES.index(crop_type)
    df = pd.DataFrame([{
        'Temparature': temp,
        'Moisture': moisture,
        'Soil Type': soil_encoded,
        'Crop Type': crop_encoded,
        'Nitrogen': nitrogen,
        'Phosphorous': phosphorous,
        'Potassium': potassium,
        'pH': ph,
        'Humidity ': humidity
    }])[FERT_FEATURES]

    pred = fert_model.predict(df)[0]
    fertilizer = FERTILIZER_MAP.get(pred, "Unknown Fertilizer")
    return {"recommended_fertilizer": fertilizer}


# ======================================================
# 4️⃣ CHATBOT ENDPOINT (RAG + GEMINI)
# ======================================================

API_KEY = os.getenv("OPENROUTER_API_KEY")

# (Optional) System Prompt

async def ask_openrouter(message: str):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer " + API_KEY,
    }

    data = {
        "model": "openai/gpt-3.5-turbo",        # you can use any model
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]
    }

    # Use thread executor because requests is blocking
    loop = asyncio.get_event_loop()

    def send_request():
        return requests.post(url, headers=headers, json=data)

    response = await loop.run_in_executor(None, send_request)

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"OpenRouter Error: {response.text}"
        )

    result = response.json()
    return result["choices"][0]["message"]["content"]


# ------------------------------------------------------------------
# ✅ FINAL WORKING CHATBOT ROUTE USING OPENROUTER
# ------------------------------------------------------------------
@app.post("/chatbot")
async def chatbot(query: str = Form(...)):
    try:
        user_query = query.strip()

        # Call OpenRouter API
        reply = await ask_openrouter(user_query)

        return {"response": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {e}")



# ======================================================
# 5️⃣ PLANT DISEASE DETECTION ENDPOINT
# ======================================================
MODEL_PATH = r"model.tflite"
CLASSES_PATH = r"class_indices.json"
IMAGE_SIZE = (224, 224)
API_KEY = os.getenv("OPENROUTER_API_KEY") # <-- replace with your OpenRouter key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Load class indices and TFLite model
try:
    with open(CLASSES_PATH, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {int(v): k for k, v in class_indices.items()}

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("✅ Disease model loaded.")
except Exception as e:
    print("❌ Error loading disease model:", e)
    interpreter, idx_to_class = None, None


# ======================================================
# 3️⃣ IMAGE PREPROCESSING
# ======================================================
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ======================================================
# 4️⃣ GROQ LEAF DETECTION (WORKING & FIXED)
# ======================================================
def detect_leaf_groq(image_bytes: bytes) -> bool:
    """
    Converts uploaded image to JPEG + Base64 before sending to Groq.
    Handles Groq Vision model properly (no dict access).
    """
    try:
        # Re-encode image to clean JPEG
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        jpeg_bytes = buffer.getvalue()

        # Convert to Base64
        img_b64 = base64.b64encode(jpeg_bytes).decode()

        # Groq Scout Vision Call
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text":
                                "Does this image contain a plant leaf? Reply strictly YES or NO."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}"
                            }
                        }
                    ],
                }
            ],
            max_completion_tokens=10,
        )

        # 🟢 FIXED PART — correct way to read output
        result = completion.choices[0].message.content.strip().upper()

        print("Groq Leaf Detection →", result)

        return "YES" in result

    except Exception as e:
        print("Groq Vision Error:", e)
        return False


# ======================================================
# 5️⃣ DISEASE PREDICTION USING TFLITE
# ======================================================
def predict_disease(interpreter, input_array, idx_to_class):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(preds))

    return idx_to_class.get(idx, "Unknown"), float(preds[idx]) * 100


# ======================================================
# 6️⃣ OPENROUTER TREATMENT RECOMMENDATION
# ======================================================
async def get_openrouter_recommendation(disease_name: str) -> str:
    prompt = (
        f"Give simple treatment steps for {disease_name} in farmer-friendly language. "
        "Use bullet points and simple wording."
    )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT },
            {"role": "user", "content": prompt},
        ],
    }

    loop = asyncio.get_event_loop()

    def send():
        return requests.post(url, headers=headers, json=data)

    try:
        response = await loop.run_in_executor(None, send)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"Recommendation error: {response.status_code}"
    except Exception as e:
        return f"Error generating recommendation: {str(e)}"


# ======================================================
# 7️⃣ FINAL API ENDPOINT
# ======================================================
@app.post("/predict-disease")
async def predict_disease_api(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Disease model not loaded.")

    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")

        image_bytes = await file.read()

        # Load image
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(400, "Invalid image file")

        # Step 1: Groq leaf detection
        leaf_present = detect_leaf_groq(image_bytes)

        if not leaf_present:
            return {
                "status": "error",
                "leaf_detected": False,
                "message": "No leaf detected. Upload a clear leaf image.",
            }

        # Step 2: Disease prediction
        input_arr = preprocess_image(image)
        disease, conf = predict_disease(interpreter, input_arr, idx_to_class)

        # Step 3: Recommendations via OpenRouter
        recommendation = await get_openrouter_recommendation(disease)

        return {
            "status": "success",
            "leaf_detected": True,
            "predicted_disease": disease,
            "confidence": f"{conf:.2f}",
            "recommendation": recommendation,
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))
# Report Upload Code 
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import fitz  # PyMuPDF
import base64
import uuid
from fpdf import FPDF
import os
import io
import re
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ---------------- Markdown → Plain Text Converter ---------------- #
def markdown_to_plain(text):
    # Remove headings (#, ##, ###, etc.)
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)
    # Remove bold (**text** or __text__)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
    # Remove italic (*text* or _text_)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    # Remove inline code `text`
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Remove links [text](url)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove images ![alt](url)
    text = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove blockquotes
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)
    # Remove bullet points
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^\s*([-*_]){3,}\s*$', '', text, flags=re.MULTILINE)
    # Clean empty lines
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text.strip()


def resize_image(img: Image.Image, max_size=(1024, 1024)):
    img.thumbnail(max_size)
    return img


def encode_pil_image(img: Image.Image):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()

    # -------- Add FarmSathi Logo (top-right corner) --------
    # Make sure logo.png is in the same folder
    if os.path.exists("farmsathi.jpg"):
        pdf.image("farmsathi.jpg", x=160, y=10, w=35)

    # -------- Title --------
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "FarmSathi - Crop Treatment Report", ln=True, align="C")
    pdf.ln(10)

    # -------- Treatment Text --------
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, text)

    pdf.output(filename)
    return filename



def convert_pdf_to_image(pdf_bytes):
    """Convert first PDF page to a PIL image using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=120)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    return img


@app.post("/upload-report")
async def upload_report(file: UploadFile = File(...)):
    try:
        # ---------------- Read File ---------------- #
        if file.content_type.startswith("image/"):
            img = Image.open(file.file)

        elif file.content_type == "application/pdf":
            pdf_bytes = await file.read()
            img = convert_pdf_to_image(pdf_bytes)

        else:
            raise HTTPException(status_code=400, detail="Upload must be PDF or Image.")

        # ---------------- Resize & Encode ---------------- #
        img = resize_image(img)
        img_b64 = encode_pil_image(img)

        # ---------------- Send to Groq ---------------- #
        prompt_text = (
            "You are an expert agronomist. I will provide you an image of a crop leaf. "
            "Analyze the image carefully and identify the disease affecting the plant. "
            "Then provide a short and clear treatment advice suitable for a farmer. "
            "Include: disease name, key symptoms, and 3–5 simple steps. "
            "Keep it farmer-friendly."
        )

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                    ],
                }
            ],
        )

        treatment_raw = response.choices[0].message.content

        # ---------------- Convert Markdown → Plain Text ---------------- #
        treatment_text = markdown_to_plain(treatment_raw)

        # ---------------- Generate Final PDF ---------------- #
        output_pdf = f"treatment_{uuid.uuid4().hex}.pdf"
        generate_pdf(treatment_text, output_pdf)

        return FileResponse(output_pdf, filename="treatment_report.pdf")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")



# ======================================================
# ✅ RUN SERVER
# ======================================================


