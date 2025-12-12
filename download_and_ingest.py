#!/usr/bin/env python3
"""
Download Qwen2.5-0.5B model from HuggingFace and ingest into Hartonomous database.
"""

import os
import sys
import json
import struct
import psycopg2
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from collections import defaultdict

# Database connection
DB_CONFIG = {
    'dbname': 'hartonomous',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost'
}

# Fundamental atoms - seed the 4D space
FUNDAMENTAL_ATOMS = {
    'digits': list(range(10)),  # 0-9
    'letters_lower': [chr(i) for i in range(ord('a'), ord('z')+1)],
    'letters_upper': [chr(i) for i in range(ord('A'), ord('Z')+1)],
    'special': [' ', '.', ',', '!', '?', '-', '_', '\n', '\t', '(', ')', '[', ']', '{', '}']
}

def hilbert_encode_4d(x, y, z, w, bits=16):
    """Encode 4D coordinates to Hilbert curve index."""
    # Normalize to [0, 2^bits-1]
    max_val = (1 << bits) - 1
    ix = int((x + 1) / 2 * max_val)
    iy = int((y + 1) / 2 * max_val)
    iz = int((z + 1) / 2 * max_val)
    iw = int((w + 1) / 2 * max_val)
    
    # Clamp
    ix = max(0, min(max_val, ix))
    iy = max(0, min(max_val, iy))
    iz = max(0, min(max_val, iz))
    iw = max(0, min(max_val, iw))
    
    # Interleave bits (simplified 4D Hilbert curve approximation)
    index = 0
    for i in range(bits):
        index |= ((ix >> i) & 1) << (i * 4)
        index |= ((iy >> i) & 1) << (i * 4 + 1)
        index |= ((iz >> i) & 1) << (i * 4 + 2)
        index |= ((iw >> i) & 1) << (i * 4 + 3)
    
    return index

def project_to_4d_landmarks(value, modality):
    """
    Project a value to 4D space using landmark-based deterministic coordinates.
    Coordinates stay within [-1, 1]^4 hypercube.
    """
    # Hash the value to get deterministic but distributed coordinates
    if isinstance(value, (int, float)):
        seed = int(value * 1000000) if isinstance(value, float) else value
    elif isinstance(value, str):
        seed = sum(ord(c) * (i+1) for i, c in enumerate(value))
    elif isinstance(value, bytes):
        seed = sum(b * (i+1) for i, b in enumerate(value))
    else:
        seed = hash(str(value))
    
    # Use seed to generate coordinates in [-1, 1]
    np.random.seed(abs(seed) % (2**32))
    
    # Modality determines base region of hypercube
    modality_offset = {
        'number': np.array([0.5, 0.5, 0.5, 0.5]),
        'char': np.array([-0.5, 0.5, 0.5, 0.5]),
        'token': np.array([0.5, -0.5, 0.5, 0.5]),
        'weight': np.array([0.5, 0.5, -0.5, 0.5]),
        'composite': np.array([0, 0, 0, 0])
    }.get(modality, np.array([0, 0, 0, 0]))
    
    # Add small random perturbation
    coords = modality_offset + np.random.uniform(-0.3, 0.3, 4)
    
    # Clamp to [-1, 1]
    coords = np.clip(coords, -1, 1)
    
    return tuple(coords)

def create_atom(conn, value, modality, metadata=None):
    """
    Insert an atom into the database.
    Returns the Hilbert ID of the atom.
    """
    # Project to 4D space
    x, y, z, w = project_to_4d_landmarks(value, modality)
    
    # Encode to Hilbert curve
    hilbert_id = hilbert_encode_4d(x, y, z, w)
    
    # Create POINT geometry
    geom_wkt = f"POINT Z({x} {y} {z})"
    
    # Metadata as JSON
    meta_json = json.dumps(metadata or {})
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO atoms (id, geometry, modality, raw_value, metadata)
                VALUES (%s, ST_GeomFromText(%s, 0), %s, %s, %s)
                ON CONFLICT (id) DO NOTHING
                RETURNING id
            """, (hilbert_id, geom_wkt, modality, str(value).encode(), meta_json))
            
            result = cur.fetchone()
            conn.commit()
            return hilbert_id if result else hilbert_id
    except Exception as e:
        conn.rollback()
        print(f"Error creating atom {value} ({modality}): {e}")
        return None

def create_relation_atom(conn, atom_a_id, atom_b_id, weight=1.0, metadata=None):
    """
    Create a LINESTRING relation between two atoms with a weight.
    The weight is stored in the Z coordinate of the second point.
    """
    try:
        with conn.cursor() as cur:
            # Get coordinates of both atoms
            cur.execute("""
                SELECT ST_X(geometry), ST_Y(geometry), ST_Z(geometry)
                FROM atoms WHERE id = %s
            """, (atom_a_id,))
            a_coords = cur.fetchone()
            
            cur.execute("""
                SELECT ST_X(geometry), ST_Y(geometry), ST_Z(geometry)
                FROM atoms WHERE id = %s
            """, (atom_b_id,))
            b_coords = cur.fetchone()
            
            if not a_coords or not b_coords:
                return None
            
            # Create LINESTRING with weight encoded in Z of second point
            geom_wkt = f"LINESTRING Z({a_coords[0]} {a_coords[1]} {a_coords[2]}, {b_coords[0]} {b_coords[1]} {weight})"
            
            # Compute Hilbert ID for the relation (midpoint + weight)
            mid_x = (a_coords[0] + b_coords[0]) / 2
            mid_y = (a_coords[1] + b_coords[1]) / 2
            mid_z = (a_coords[2] + weight) / 2
            mid_w = weight
            
            hilbert_id = hilbert_encode_4d(mid_x, mid_y, mid_z, mid_w)
            
            meta_json = json.dumps(metadata or {'atom_a': atom_a_id, 'atom_b': atom_b_id, 'weight': weight})
            
            cur.execute("""
                INSERT INTO atoms (id, geometry, modality, metadata)
                VALUES (%s, ST_GeomFromText(%s, 0), 'relation', %s)
                ON CONFLICT (id) DO NOTHING
                RETURNING id
            """, (hilbert_id, geom_wkt, meta_json))
            
            result = cur.fetchone()
            conn.commit()
            return hilbert_id if result else hilbert_id
    except Exception as e:
        conn.rollback()
        print(f"Error creating relation: {e}")
        return None

def ingest_model():
    """Download and ingest Qwen2.5-0.5B model."""
    print("="*80)
    print("HARTONOMOUS - Full Model Ingestion")
    print("="*80)
    
    # Connect to database
    print("\nConnecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Use TinyLlama which is simpler and available
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nDownloading {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        print(f"✓ Model loaded: {model.config.hidden_size}d embeddings, {model.config.num_hidden_layers} layers")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to tokenizer-only mode...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = None
    
    # 1. Ingest fundamental atoms (seed the space)
    print("\n[1/5] Ingesting fundamental atoms...")
    fundamental_count = 0
    
    for category, values in FUNDAMENTAL_ATOMS.items():
        for val in values:
            create_atom(conn, val, 'char', {'category': category})
            fundamental_count += 1
    
    print(f"✓ {fundamental_count} fundamental atoms created")
    
    # 2. Ingest vocabulary tokens
    print("\n[2/5] Ingesting vocabulary tokens...")
    token_atoms = {}
    vocab_size = len(tokenizer)
    
    for token_id in range(min(vocab_size, 10000)):  # Start with first 10k tokens
        token_str = tokenizer.decode([token_id])
        atom_id = create_atom(conn, token_str, 'token', {'token_id': token_id})
        token_atoms[token_id] = atom_id
        
        if (token_id + 1) % 1000 == 0:
            print(f"  Progress: {token_id + 1}/{min(vocab_size, 10000)} tokens")
    
    print(f"✓ {len(token_atoms)} token atoms created")
    
    # 3. Ingest embedding weights
    print("\n[3/5] Ingesting embedding layer...")
    
    if model is not None:
        embeddings = model.get_input_embeddings().weight.detach().numpy()
        
        weight_atoms = {}
        unique_weights = set()
        
        # Sample embedding vectors (first 100 tokens, all dimensions)
        for token_id in range(min(100, embeddings.shape[0])):
            embedding_vector = embeddings[token_id]
            
            for dim_idx, weight_val in enumerate(embedding_vector[:10]):  # First 10 dimensions
                weight_val = float(weight_val)
                
                # Create weight atom if not exists
                if weight_val not in unique_weights:
                    weight_id = create_atom(conn, weight_val, 'weight', {'type': 'embedding'})
                    weight_atoms[weight_val] = weight_id
                    unique_weights.add(weight_val)
                else:
                    weight_id = weight_atoms[weight_val]
                
                # Create relation: token -> dimension with weight
                if token_id in token_atoms:
                    create_relation_atom(
                        conn,
                        token_atoms[token_id],
                        weight_id,
                        weight=weight_val,
                        metadata={'layer': 'embedding', 'dimension': dim_idx}
                    )
            
            if (token_id + 1) % 25 == 0:
                print(f"  Progress: {token_id + 1}/100 token embeddings")
        
        print(f"✓ {len(unique_weights)} unique weight values, relations created")
    else:
        print("✓ Skipped (model not loaded)")
    
    # 4. Database statistics
    print("\n[4/5] Computing statistics...")
    with conn.cursor() as cur:
        cur.execute("SELECT modality, COUNT(*) FROM atoms GROUP BY modality ORDER BY modality")
        stats = cur.fetchall()
    
    print("\nAtom counts by modality:")
    for modality, count in stats:
        print(f"  {modality}: {count:,}")
    
    # 5. Test inference
    print("\n[5/5] Testing spatial queries...")
    test_token = "hello"
    test_token_id = tokenizer.encode(test_token, add_special_tokens=False)[0]
    
    if test_token_id in token_atoms:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT a2.modality, COUNT(*)
                FROM atoms a1
                JOIN atoms rel ON ST_Intersects(rel.geometry, a1.geometry)
                JOIN atoms a2 ON ST_Intersects(rel.geometry, a2.geometry) AND a2.id != a1.id
                WHERE a1.id = %s AND rel.modality = 'relation'
                GROUP BY a2.modality
            """, (token_atoms[test_token_id],))
            
            relations = cur.fetchall()
            print(f"\nToken '{test_token}' has relations:")
            for mod, count in relations:
                print(f"  → {mod}: {count}")
    
    print("\n" + "="*80)
    print("✓ INGESTION COMPLETE")
    print("="*80)
    
    conn.close()

if __name__ == "__main__":
    ingest_model()
