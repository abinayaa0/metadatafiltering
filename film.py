import torch
import torch.nn as nn
import pandas as pd
import json
import psycopg2
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModel


# ========== CONFIGURATION ==========
EMBEDDING_DIM = 384  # For "all-MiniLM-L6-v2"
MODEL_NAME = "allenai/scibert_scivocab_uncased"
EMBEDDING_DIM = 768 
DB_URI = 'postgresql://postgres:12345@localhost:5432/arxivDB'  # Update this
TABLE_NAME = 'arxiv_papers'
FILE_PATH = r"C:\\Users\\Abinaya\\Downloads\\archive (3)\\arxiv-metadata-oai-snapshot.json"

# ========== FILM GATING MODEL ==========
class FiLMGate(nn.Module):
    def __init__(self, meta_dim, text_dim):
        super().__init__()
        self.gamma = nn.Linear(meta_dim, text_dim)
        self.beta = nn.Linear(meta_dim, text_dim)

    def forward(self, text_emb, meta_emb):
        gamma = self.gamma(meta_emb)
        beta = self.beta(meta_emb)
        return gamma * text_emb + beta

# ========== STEP 1: LOAD AND PARSE JSON ==========
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# ========== STEP 2: TEXT + METADATA EMBEDDING ==========
def get_text_embedding(tokenizer, model, text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        return cls_embedding.squeeze(0)


def get_metadata_embedding(paper):
    # Define known metadata values for one-hot encoding
    cat_labels = ['hep-ph', 'math.', 'physics.gen-ph', 'cs.LG', 'astro-ph']
    submitters = ['Pan', 'Streinu', 'Theran']
    journals = ['PhysRevD', 'Nature', 'Science']
    flags = ['doi', 'comments']

    total_dim = len(cat_labels) + len(submitters) + len(journals) + len(flags)
    vec = torch.zeros(total_dim)

    idx = 0
    cats = paper.get('categories', '')
    for cat in cat_labels:
        if cat in cats:
            vec[idx] = 1
        idx += 1

    submitter = paper.get('submitter', '')
    for s in submitters:
        if s in submitter:
            vec[idx] = 1
        idx += 1

    journal = paper.get('journal-ref') or ''
    for j in journals:
        if j in journal:
            vec[idx] = 1
        idx += 1

    doi = paper.get('doi')
    comments = paper.get('comments')
    vec[idx] = 1 if doi else 0
    idx += 1
    vec[idx] = 1 if comments else 0

    return vec

# ========== STEP 3: CREATE FUSED EMBEDDINGS ==========
def generate_fused_embeddings(data):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    gate = FiLMGate(meta_dim=13, text_dim=EMBEDDING_DIM)

    results = []
    for paper in data:
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        text = title + ' ' + abstract

        text_emb = get_text_embedding(tokenizer, model, text)
        meta_emb = get_metadata_embedding(paper)
        fused_emb = gate(text_emb, meta_emb).detach().numpy()

        results.append({
            'id': paper['id'],
            'title': title,
            'embedding': fused_emb.tolist(),
        })
    return results


# ========== STEP 4: STORE IN PGVECTOR ==========
def store_pgvector(results):
    import pgvector.psycopg2
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id TEXT PRIMARY KEY,
            title TEXT,
            embedding VECTOR({EMBEDDING_DIM})
        );
    """)

    for row in results:
        cur.execute(f"""
            INSERT INTO {TABLE_NAME} (id, title, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (row['id'], row['title'], row['embedding']))

    conn.commit()
    conn.close()

# ========== MAIN ==========
if __name__ == '__main__':
    data = load_data(FILE_PATH)[:500]  # slice to test first
    results = generate_fused_embeddings(data)
    store_pgvector(results)
    print("Stored fused embeddings with FiLM gating to pgvector.")
