import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sciBERT import df_preprocessed  # Make sure df_preprocessed contains 'id', 'text', and 'embedding'

# Fix the SettingWithCopyWarning
df_preprocessed = df_preprocessed.copy()

# Update with your own DB credentials
DB_URL = "postgresql://postgres:12345@localhost:5432/arxivDB"
TABLE_NAME = "arxiv_sci_bert_embeddings"
EMBEDDING_DIM = 768  # SciBERT outputs 768-dim vectors

# Create the engine
engine = create_engine(DB_URL)

# Run everything in one transaction and one connection
with engine.begin() as conn:
    # Enable the pgvector extension
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    # Create the table if it doesn't exist
    conn.execute(text(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id TEXT PRIMARY KEY,
            text TEXT,
            embedding vector({EMBEDDING_DIM})
        );
    """))

    # Insert rows from the dataframe
    for _, row in df_preprocessed.iterrows():
        embedding = row["embedding"]
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()  # Make sure it's a list for psycopg2

        conn.execute(text(f"""
            INSERT INTO {TABLE_NAME} (id, text, embedding)
            VALUES (:id, :text, :embedding)
            ON CONFLICT (id) DO NOTHING;
        """), {
            "id": row["id"],
            "text": row["text"],
            "embedding": embedding
        })
