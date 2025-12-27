import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords
nltk.download("stopwords")

ps = PorterStemmer()

def load_dataset(path):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(path, encoding="latin-1")
    return df

def clean_columns(df):
    print("[INFO] Cleaning unnecessary columns and renaming...")
    df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
    df = df.rename(columns={"v1": "Category", "v2": "Message"})
    return df

def undersample(df):
    print("[INFO] Performing undersampling...")

    spam = df[df["Category"] == "spam"]
    ham = df[df["Category"] == "ham"].sample(len(spam), random_state=42)

    df_balanced = pd.concat([spam, ham], axis=0)
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

    df_balanced["Label"] = df_balanced["Category"].map({"ham": 0, "spam": 1})
    return df_balanced

def clean_text(msg):
    msg = re.sub("[^a-zA-Z]", " ", msg)
    msg = msg.lower()
    words = msg.split()
    words = [ps.stem(w) for w in words if w not in stopwords.words("english")]
    return " ".join(words)

def main():
    df = load_dataset("sms_dataset.csv")
    df = clean_columns(df)
    df_balanced = undersample(df)

    print("[INFO] Applying text cleaning...")
    df_balanced["Clean"] = df_balanced["Message"].apply(clean_text)

    df_balanced.to_csv("processed_spam_dataset.csv", index=False)
    print("[INFO] Saved clean dataset → processed_spam_dataset.csv")

if __name__ == "__main__":
    main()
