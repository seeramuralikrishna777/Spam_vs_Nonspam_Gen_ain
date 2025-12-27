from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import re
import numpy as np
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# ----------------------------------------------------
# Settings
# ----------------------------------------------------
MODEL_PATH = "spam_lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 200

app = FastAPI(title="Spam Detector (LSTM)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

try:
    _ = stopwords.words("english")
except:
    nltk.download("stopwords")

ps = PorterStemmer()

# ----------------------------------------------------
# Load Model + Tokenizer
# ----------------------------------------------------
model = load_model(MODEL_PATH)
tokenizer = pickle.load(open(TOKENIZER_PATH, "rb"))

# ----------------------------------------------------
# Text Clean + Encode (same as training)
# ----------------------------------------------------
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stopwords.words("english")]
    return " ".join(words)

def encode(text):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=MAX_LEN)

# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: dict):
    txt = data.get("message", "")
    clean = clean_text(txt)
    padded = encode(clean)
    prob = float(model.predict(padded)[0][0])
    label = "SPAM" if prob > 0.5 else "HAM"
    return {"prediction": label, "probability": prob}

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request, message: str = Form(...)):
    clean = clean_text(message)
    padded = encode(clean)
    prob = float(model.predict(padded)[0][0])
    label = "SPAM" if prob > 0.5 else "HAM"
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "message": message, "prediction": label, "probability": prob}
    )
