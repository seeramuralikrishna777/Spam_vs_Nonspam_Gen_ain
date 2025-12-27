📩 Spam Detector (LSTM) Author: Praveen Arella A simple LSTM-based spam classifier with a FastAPI web interface (HTML UI + JSON API). This project demonstrates the full ML pipeline — preprocessing → tokenization → LSTM model training → model export → production inference through FastAPI.

📁 Project Structure project/ │── app.py # FastAPI app (UI + JSON API) │── lstm_training.py # Model training script │── tokenizer.pkl # Saved tokenizer │── spam_lstm_model_tf/ # Saved model directory (or spam_lstm_model.h5) │── templates/ │ └── index.html # Web UI │── static/ # CSS / JS │── requirements.txt # Dependencies │── README.md # Documentation

🔍 Key Notes

clean_text() → stopwords + stemming.

encode() → loads tokenizer.pkl and applies pad_sequences(MAX_LEN).

API Endpoints:

GET / → Web UI

POST /predict → JSON API

POST /predict_form → HTML form submission

📊 Model Metrics Training: 6 epochs ✔ Training Summary

Final Training Accuracy: 0.9966

Final Validation Accuracy: ~0.9749

✔ Test Set Evaluation

Accuracy: 0.9766

✔ Classification Report ClassPrecisionRecallF1-scoreSupportHAM (0)0.970.980.98149SPAM (1)0.980.970.98150 ➡ Macro F1-score ~0.98 ✔ Interpretation

High precision → very few ham misclassified as spam

High recall → spam is rarely missed

Balanced F1 → overall strong model performance

✔ Recommended Validations

Ensure no data leakage

Use stratified train-test split

Consider k-fold cross-validation

Test with real-world messages

▶ How to Run (Windows / PowerShell) 1️⃣ Create Virtual Environment python -m venv .venv ..venv\Scripts\Activate python -m pip install --upgrade pip

2️⃣ Install Dependencies pip install -r requirements.txt

3️⃣ Download NLTK Stopwords python -m nltk.downloader stopwords

4️⃣ Train the Model python lstm_training.py

This will generate:

tokenizer.pkl

spam_lstm_model_tf/ or spam_lstm_model.h5

5️⃣ Start FastAPI Server uvicorn app:app --reload --host 0.0.0.0 --port 8000

Web UI: http://127.0.0.1:8000

JSON API: POST → http://127.0.0.1:8000/predict

🧪 Example API Request (cURL) curl -X POST "http://127.0.0.1:8000/predict" ^ -H "Content-Type: application/json" ^ -d "{"message": "Congratulations! You've won a prize"}"

⚠ Troubleshooting ❗ 1. ValueError: Unrecognized keyword arguments passed to LSTM: {'time_major': False} Cause: Mismatch between TensorFlow version during training vs inference. Quick Fix (in app.py): Wrap model loading with compatibility layers. Long-Term Fix: Re-train and save using the new format: model.save("spam_lstm_model.keras")

Also: pin TensorFlow version in requirements.txt.

❗ 2. NLTK Stopwords Error Run: python -m nltk.downloader stopwords

🔒 Production Considerations 📦 Environment

Pin versions in requirements.txt

Optional: build Dockerfile

📦 Model Lifecycle

Version models (model_v1, model_v2…)

Add metadata endpoint

🔍 Testing

Unit tests for preprocessing + tokenization

Integration test with full inference

🔐 Security

Validate input JSON

Restrict CORS

Sanitize user messages

🚀 Future Improvements

Add real-world dataset for robustness

Perform k-fold validation

Try Transformer-based models

Add ROC-AUC, PR curves

Add confusion matrix visualization

📜 License Choose and include a license (MIT recommended).
