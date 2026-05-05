import re
import json
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords


def load_real_customer_data(csv_path: Path):
    raw_df = pd.read_csv(csv_path)
    description = raw_df['Ticket Description'].astype(str).str.strip()
    df_local = pd.DataFrame({
        'ticket_id': [f'TICKET_{i:04d}' for i in range(len(raw_df))],
        'description': description
    })
    return df_local[df_local['description'].ne('')].reset_index(drop=True)


def split_template_and_free_text(text):
    text = str(text)
    entities = re.findall(r'\{[^{}]+\}', text)
    free_text = re.sub(r'\{[^{}]+\}', ' ', text)
    free_text = re.sub(r'\s+', ' ', free_text).strip()
    return entities, free_text

ENGLISH_STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z]', ' ', text)
    tokens = [w for w in text.split() if w not in ENGLISH_STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

def save_json(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {filepath}")

def save_csv(df, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"Saved: {filepath}")