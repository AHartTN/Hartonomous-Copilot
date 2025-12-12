#!/usr/bin/env python3
"""
Build neural network connections in the Hartonomous database.
Uses existing atoms and creates weight-based connections between them.
"""

import psycopg2
import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

DB_CONFIG = {
    'dbname': 'hartonomous',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost'
}

def build_connections():
    print("="*80)
    print("HARTONOMOUS - Building Neural Network Connections")
    print("="*80)
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"✓ Loaded tokenizer with {len(tokenizer)} tokens")
    
    # Get existing atoms
    print("\n[2/4] Mapping existing atoms...")
    with conn.cursor() as cur:
        # Map tokens to their atom IDs
        cur.execute("""
            SELECT id, raw_value
            FROM atoms
            WHERE raw_value IS NOT NULL
            ORDER BY id
        """)
        
        atom_map = {}  # raw_value -> atom_id
        for atom_id, raw_value in cur.fetchall():
            atom_map[bytes(raw_value)] = atom_id
    
    print(f"✓ Found {len(atom_map)} existing atoms")
    
    # Create weight atoms for common weight values
    print("\n[3/4] Creating weight atoms...")
    
    # Common weight values in quantized models
    weight_values = np.linspace(-1.0, 1.0, 256)  # 256 quantization levels
    weight_atoms = {}
    
    with conn.cursor() as cur:
        for i, weight in enumerate(weight_values):
            weight_bytes = str(weight).encode()
            
            if weight_bytes not in atom_map:
                # Create POINT ZM for weight value
                # Use deterministic coordinates based on weight value
                x = weight  # X coordinate = the weight itself
                y = 0.0
                z = abs(weight)  # Z = magnitude
                m = float(i) / 255.0  # M = normalized index
                
                # Simple Hilbert-like ID (weight-based)
                hilbert_id = hash(weight) & 0x7FFFFFFF  # Positive 32-bit hash
                
                cur.execute("""
                    INSERT INTO atoms (id, geom, raw_value)
                    VALUES (%s, ST_MakePoint(%s, %s, %s, %s), %s)
                    ON CONFLICT (id) DO NOTHING
                    RETURNING id
                """, (hilbert_id, float(x), float(y), float(z), float(m), weight_bytes))
                
                result = cur.fetchone()
                if result:
                    weight_atoms[weight] = hilbert_id
                    atom_map[weight_bytes] = hilbert_id
                else:
                    # Already exists, fetch it
                    cur.execute("SELECT id FROM atoms WHERE raw_value = %s", (weight_bytes,))
                    existing = cur.fetchone()
                    if existing:
                        weight_atoms[weight] = existing[0]
                        atom_map[weight_bytes] = existing[0]
            else:
                weight_atoms[weight] = atom_map[weight_bytes]
            
            if (i + 1) % 50 == 0:
                conn.commit()
                print(f"  Progress: {i+1}/256 weight values")
    
    conn.commit()
    print(f"✓ {len(weight_atoms)} weight atoms ready")
    
    # Build connections between tokens using co-occurrence
    print("\n[4/4] Building token connections...")
    
    # Sample text for demonstration
    sample_texts = [
        "Hello world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence and machine learning are transforming technology.",
        "PostgreSQL is a powerful database system.",
        "Python is great for data science and AI."
    ]
    
    connection_count = 0
    
    with conn.cursor() as cur:
        for text in sample_texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            
            # Create bigram connections (token[i] -> token[i+1])
            for i in range(len(tokens) - 1):
                token_a = tokens[i]
                token_b = tokens[i + 1]
                
                # Get token strings
                str_a = tokenizer.decode([token_a]).encode()
                str_b = tokenizer.decode([token_b]).encode()
                
                # Find atom IDs
                atom_a_id = atom_map.get(str_a)
                atom_b_id = atom_map.get(str_b)
                
                if atom_a_id and atom_b_id:
                    # Use a simple weight (could be learned from actual model)
                    weight_val = 0.5 + (i / len(tokens)) * 0.5  # Gradually increasing weights
                    
                    # Find nearest quantized weight
                    weight_idx = int((weight_val + 1) / 2 * 255)
                    quantized_weight = weight_values[weight_idx]
                    weight_atom_id = weight_atoms.get(quantized_weight)
                    
                    if weight_atom_id:
                        # Insert connection
                        cur.execute("""
                            INSERT INTO connections (from_id, to_id, weight_id, layer)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (atom_a_id, atom_b_id, weight_atom_id, 'embedding'))
                        
                        connection_count += 1
            
            conn.commit()
            print(f"  Processed: '{text[:50]}...' ({len(tokens)} tokens)")
    
    print(f"✓ Created {connection_count} connections")
    
    # Statistics
    print("\n" + "="*80)
    print("DATABASE STATISTICS")
    print("="*80)
    
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM atoms")
        atom_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM connections")
        conn_count = cur.fetchone()[0]
        
        print(f"Total atoms: {atom_count:,}")
        print(f"Total connections: {conn_count:,}")
        print(f"Average connections per atom: {conn_count/atom_count:.2f}")
    
    conn.close()
    
    print("\n✓ CONNECTION BUILDING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    build_connections()
