#!/usr/bin/env python3
import subprocess
import psycopg2
import psycopg2.extras
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

print("Downloading Qwen2.5-0.5B...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)

print(f"Model loaded: {model.config}")
print(f"Vocab size: {len(tokenizer)}")

# Get embedding weights
embed_weights = model.embed_tokens.weight.detach().numpy()
print(f"Embedding shape: {embed_weights.shape}")

# Ingest sample
conn = psycopg2.connect("dbname=hartonomous user=postgres")
cur = conn.cursor()

# Sample: first 1000 tokens, first 50 dimensions
sample_size = 1000
dim_size = 50

batch = []
for tok_id in range(sample_size):
    for dim in range(dim_size):
        weight = float(embed_weights[tok_id, dim])
        
        # Simple 4D projection
        x = np.tanh(weight)
        y = dim / dim_size
        z = tok_id / sample_size
        w = weight / 10.0
        
        # Hilbert encode (simple interleave)
        hid = (tok_id * 1000000) + dim
        
        batch.append((
            hid,
            f'POINT({x} {y} {z} {w})',
            weight.tobytes()
        ))
        
        if len(batch) >= 5000:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO atoms (id, geometry, raw_value) VALUES %s ON CONFLICT DO NOTHING",
                batch,
                template="(%s, ST_GeomFromText(%s, 0), %s)"
            )
            batch = []
            print(f"Inserted up to token {tok_id}, dim {dim}")

if batch:
    psycopg2.extras.execute_values(
        cur,
        "INSERT INTO atoms (id, geometry, raw_value) VALUES %s ON CONFLICT DO NOTHING",
        batch,
        template="(%s, ST_GeomFromText(%s, 0), %s)"
    )

conn.commit()
cur.close()
conn.close()

print(f"Ingested {sample_size * dim_size} weight atoms!")
