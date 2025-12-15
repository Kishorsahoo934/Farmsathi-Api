# FarmSathi Backend (FastAPI)

A production-ready FastAPI backend for agriculture assistance that powers:

- Crop recommendation
- Fertilizer recommendation
- Plant disease detection (TFLite)
- Gemini-powered chatbot (concise, farmer-friendly responses)

This README is tailored for the backend only.

## Tech Stack

- FastAPI + Uvicorn
- TensorFlow Lite (image inference)
- scikit-learn / joblib (ML models)
- Google Gemini (`google-generativeai`)
- NumPy, Pandas, Pillow

## Project Structure

```
cropdiseaseprediction_final/
├─ app.py                      # FastAPI application entrypoint
├─ requirements.txt            # Python dependencies
├─ crop_Recommendation_random_forest.pkl
├─ fertilizer_recommendation_model_latest.joblib
├─ fertilizer_model_metadata_latest.json
├─ Crop_Recommendation_Dataset.csv
├─ (expected) model.tflite                # TFLite disease model (place here)
├─ (expected) class_indices.json          # Class index mapping (place here)
└─ README.md                   # This file
```

## Prerequisites

- Python 3.9+ recommended
- pip

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure the Google Gemini API key.

- Preferred: set an environment variable and update `app.py` to read from it.
- Current code configures directly in `app.py` with `genai.configure(api_key=...)`. Replace this with your key, or better, load from env:

```python
# Example (recommended change in app.py):
# import os
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
```

- Set the env var (PowerShell):

```powershell
$env:GOOGLE_API_KEY = "<your_gemini_api_key>"
```

4. Ensure required model/artifact files exist alongside `app.py`:

- `crop_Recommendation_random_forest.pkl` (already present)
- `fertilizer_recommendation_model_latest.joblib` (already present)
- `fertilizer_model_metadata_latest.json` (already present)
- `model.tflite` (you must add this file)
- `class_indices.json` (you must add this file)

## Run the Server

```bash
python app.py
```

- Runs on `http://0.0.0.0:8000`
- CORS is enabled for all origins (adjust for production).
- Open API docs: `http://localhost:8000/docs`

## API Endpoints

### 1) Crop Recommendation

- URL: `POST /crop-recommend`
- Body: `multipart/form-data`
- Fields:
  - `nitrogen` (float)
  - `phosphorus` (float)
  - `potassium` (float)
  - `temperature` (float)
  - `humidity` (float)
  - `ph` (float)
  - `rainfall` (float)
- Response:

```json
{
  "recommended_crop": "rice"
}
```

- cURL:

```bash
curl -X POST http://localhost:8000/crop-recommend \
  -F nitrogen=90 -F phosphorus=40 -F potassium=40 \
  -F temperature=25.5 -F humidity=70 -F ph=6.5 -F rainfall=200
```

### 2) Fertilizer Recommendation

- URL: `POST /fertilizer-recommend`
- Body: `multipart/form-data`
- Fields:
  - `temp` (float)
  - `humidity` (float)
  - `moisture` (float)
  - `nitrogen` (float)
  - `phosphorous` (float)
  - `potassium` (float)
  - `ph` (float)
  - `soil_type` (str) one of: `Sandy`, `Loamy`, `Black`, `Red`, `Clayey`
  - `crop_type` (str) one of: `Maize`, `Sugarcane`, `Cotton`, `Tobacco`, `Paddy`, `Barley`, `Wheat`, `Millets`, `Oil seeds`, `Pulses`, `Ground Nuts`
- Response:

```json
{
  "recommended_fertilizer": "Urea"
}
```

- cURL:

```bash
curl -X POST http://localhost:8000/fertilizer-recommend \
  -F temp=26 -F humidity=65 -F moisture=40 \
  -F nitrogen=40 -F phosphorous=36 -F potassium=20 -F ph=6.8 \
  -F soil_type=Loamy -F crop_type=Wheat
```

### 3) Chatbot (Gemini)

- URL: `POST /chatbot`
- Body: `multipart/form-data`
- Fields:
  - `query` (str)
- Response:

```json
{
  "response": "... concise, bullet-point tips ..."
}
```

- cURL:

```bash
curl -X POST http://localhost:8000/chatbot -F "query=How to manage leaf rust in maize?"
```

### 4) Plant Disease Detection

- URL: `POST /predict-disease`
- Body: `multipart/form-data`
- Fields:
  - `file` (image file: jpg/png)
- Requirements:
  - `model.tflite` and `class_indices.json` must be present in the project root.
- Response:

```json
{
  "predicted_disease": "Corn Common Rust",
  "confidence": "87.12",
  "recommendation": "- bullet point steps ..."
}
```

- cURL:

```bash
curl -X POST http://localhost:8000/predict-disease \
  -F file=@/path/to/leaf.jpg
```

## Notes & Best Practices

- Replace any hardcoded API keys in `app.py` with environment variables before committing to GitHub.
- CORS is wide open for development; restrict `allow_origins` for production.
- Keep model files (`*.pkl`, `*.joblib`, `*.tflite`, metadata) out of version control if they are large; consider using Git LFS.
- The dataset CSV and HTML/CSS assets are not required to run the API; they appear to be from a demo UI.

## Troubleshooting

- Missing model files: ensure `model.tflite` and `class_indices.json` are present.
- TensorFlow errors on Windows: verify compatible Python/TensorFlow versions.
- Gemini errors: confirm `GOOGLE_API_KEY` is valid and network access is available.

## License

Add your chosen license (e.g., MIT) before open sourcing.
