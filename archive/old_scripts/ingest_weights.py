#!/usr/bin/env python3
"""
Ingest actual model weights as geometric relations.
"""

import psycopg2
import psycopg2.extras
import numpy as np
import torch
from safetensors import safe_open
import struct
from pathlib import Path

def hilbert_encode_4d(x, y, z, w, bits=16):
    """Encode 4D point to Hilbert curve index."""
    def interleave_bits(*coords):
        result = 0
        for bit in range(bits):
            for coord in coords:
                result = (result << 1) | ((coord >> bit) & 1)
        return result
    
    scale = (1 << bits) - 1
    ix = int((x + 1) / 2 * scale)
    iy = int((y + 1) / 2 * scale)
    iz = int((z + 1) / 2 * scale)
    iw = int((w + 1) / 2 * scale)
    return interleave_bits(ix, iy, iz, iw)

def get_4d_coords(value, modality, index=0, total=1):
    """Project value to 4D space deterministically."""
    x = np.tanh(float(value))
    y = float(modality) / 10.0
    z = np.sin(index / max(1, total) * np.pi)
    w = np.cos(index / max(1, total) * np.pi)
    return x, y, z, w

def ingest_embedding_layer(cur, tensor_name, tensor_data):
    """Ingest embedding layer."""
    print(f"Ingesting {tensor_name}: {tensor_data.shape}")
    
    vocab_size, embed_dim = tensor_data.shape
    batch_values = []
    
    for token_id in range(min(vocab_size, 1000)):  # Start with first 1000
        embedding_vector = tensor_data[token_id]
        
        for dim_idx in range(min(embed_dim, 100)):  # First 100 dims
            weight_value = float(embedding_vector[dim_idx])
            
            if abs(weight_value) < 1e-6:
                continue
                
            x, y, z, w = get_4d_coords(weight_value, 3, dim_idx, embed_dim)
            h_id = hilbert_encode_4d(x, y, z, w)
            
            batch_values.append((
                h_id,
                f'POINT({x} {y} {z} {w})',
                psycopg2.Binary(struct.pack('f', weight_value))
            ))
            
            if len(batch_values) >= 1000:
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO atoms (id, geometry, raw_value) VALUES %s ON CONFLICT (id) DO NOTHING",
                    batch_values,
                    template="(%s, ST_GeomFromText(%s, 0), %s)"
                )
                batch_values = []
        
        if token_id % 100 == 0:
            print(f"  Processed {token_id} tokens")
    
    if batch_values:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO atoms (id, geometry, raw_value) VALUES %s ON CONFLICT (id) DO NOTHING",
            batch_values,
            template="(%s, ST_GeomFromText(%s, 0), %s)"
        )

def main():
    safetensors_path = Path("/data/models/model.safetensors")
    
    if not safetensors_path.exists():
        print("Converting GGUF to safetensors...")
        import subprocess
        subprocess.run([
            "python3", "-c",
            """
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model.save_pretrained("/data/models/", safe_serialization=True)
"""
        ])
    
    if safetensors_path.exists():
        conn = psycopg2.connect("dbname=hartonomous user=postgres")
        cur = conn.cursor()
        
        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            for key in list(f.keys())[:5]:  # First 5 layers
                tensor = f.get_tensor(key)
                print(f"Processing {key}: {tensor.shape}")
                
                if 'embed' in key.lower() and len(tensor.shape) == 2:
                    ingest_embedding_layer(cur, key, tensor.numpy())
                    conn.commit()
        
        cur.close()
        conn.close()
        print("Done!")

if __name__ == '__main__':
    main()
