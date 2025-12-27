import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 0. NLTK Downloads
# ----------------------------------------------------
nltk.download("stopwords")
ps = PorterStemmer()

# ----------------------------------------------------
# 1. Load Raw Dataset
# ----------------------------------------------------
print("[INFO] Loading raw dataset...")
df = pd.read_csv("sms_dataset.csv", encoding="latin-1")

df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])
df = df.rename(columns={"v1": "Category", "v2": "Message"})

# ----------------------------------------------------
# 2. Balance Dataset (Best for your project)
# ----------------------------------------------------
print("[INFO] Balancing dataset...")
spam = df[df["Category"] == "spam"]
ham = df[df["Category"] == "ham"].sample(len(spam), random_state=42)

df_balanced = pd.concat([spam, ham], axis=0).sample(frac=1).reset_index(drop=True)
df_balanced["Label"] = df_balanced["Category"].map({"ham": 0, "spam": 1})

# ----------------------------------------------------
# 3. Clean Text (REAL preprocessing)
# ----------------------------------------------------
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stopwords.words("english")]
    return " ".join(words)

print("[INFO] Cleaning text...")
df_balanced["Clean"] = df_balanced["Message"].apply(clean_text)

# ----------------------------------------------------
# 4. Tokenizer + Pad Sequences
# ----------------------------------------------------
MAX_WORDS = 10000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df_balanced["Clean"])

X = tokenizer.texts_to_sequences(df_balanced["Clean"])
X = pad_sequences(X, maxlen=MAX_LEN)

y = df_balanced["Label"].values

# ----------------------------------------------------
# 5. Train-Test Split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# 6. Build LSTM Model
# ----------------------------------------------------
print("[INFO] Building model...")

model = Sequential()
model.add(Embedding(MAX_WORDS, 32, input_length=MAX_LEN))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# ----------------------------------------------------
# 7. Train
# ----------------------------------------------------
print("[INFO] Training model...")
history = model.fit(
    X_train, y_train,
    epochs=6,
    batch_size=32,
    validation_split=0.2
)

# ----------------------------------------------------
# 8. Evaluate
# ----------------------------------------------------
print("[INFO] Evaluating...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ----------------------------------------------------
# 9. Save Model + Tokenizer
# ----------------------------------------------------
model.save("spam_lstm_model.h5")
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

print("[INFO] Model and tokenizer saved!")
