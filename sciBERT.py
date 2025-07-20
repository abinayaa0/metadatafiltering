#load the data and choose a subset
import json
import random
import pandas as pd

FILE_PATH = r"C:\Users\Abinaya\Downloads\archive (3)\arxiv-metadata-oai-snapshot.json"
SAMPLE_SIZE = 1000
TARGET_CATEGORIES = [] #empty for random categories

def load_arxiv_metadata(file_path, sample_size=1000, target_categories=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if target_categories:
        filtered = []
        for line in lines:
            try:
                paper = json.loads(line)
                if any(cat in paper.get("categories", "") for cat in target_categories):
                    filtered.append(paper)
            except json.JSONDecodeError:
                continue
        selected = random.sample(filtered, min(sample_size, len(filtered)))
    else:
        selected = random.sample(lines, sample_size)
        selected = [json.loads(line) for line in selected]

    return pd.DataFrame(selected)
print("The categories of the sample choosen:")
df = load_arxiv_metadata(FILE_PATH, sample_size=SAMPLE_SIZE, target_categories=TARGET_CATEGORIES)
print(df['categories'].value_counts().head())
print("Shape of the dataset:",df.shape)
#note: The rest of the categories of the dataset are not counted because they belong to multiple categories at once.

#Preprocessing text for encoding:
def preprocess_metadata(df):
    df['title'] = df['title'].fillna('')
    df['abstract'] = df['abstract'].fillna('')
    df['categories'] = df['categories'].fillna('')
    df['authors'] = df['authors'].fillna('')
    df['comments'] = df['comments'].fillna('')
    df['journal-ref'] = df['journal-ref'].fillna('')


    df['text'] = df['title'] + ' ' + df['abstract'] + ' ' + df['categories'] + ' ' + df['authors'] + ' ' + df['comments'] + ' ' + df['journal-ref']
    return df[['id', 'text']]

df_preprocessed = preprocess_metadata(df)
print(df_preprocessed.head())

#Encoding with SciBERT:
from transformers import AutoTokenizer, AutoModel
import torch


# Load SciBERT
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")
model.eval()


def embed_text(texts):
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)
    return embeddings

df_preprocessed['embedding'] = embed_text(df_preprocessed['text'].tolist())
