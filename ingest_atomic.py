#!/usr/bin/env python3
"""
PROPER ATOMIC INGESTION
- Decompose everything to base atoms (bytes, numbers, characters)
- Build composites using LINESTRING/POLYGON with M-coordinates as references
- Use Hilbert curves for deterministic IDs
- Content-pair encoding for deduplication
"""

import json
import struct
import hashlib
import psycopg2
from pathlib import Path
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

# Hilbert curve for 4D space -> 64-bit ID
hilbert = HilbertCurve(16, 4)  # 16 bits per dimension = 64-bit output

def hilbert_id(x, y, z, m):
    """Convert 4D coordinates to Hilbert curve index"""
    coords = [
        int(x) & 0xFFFF,
        int(y) & 0xFFFF,
        int(z) & 0xFFFF,
        int(m) & 0xFFFF
    ]
    return hilbert.distance_from_point(coords)

def ingest_byte_atom(conn, byte_value):
    """Ingest a single byte as a point atom"""
    # X = byte value itself
    # Y = modality (0 = raw byte)
    # Z = 0 (unused for primitives)
    # M = 0 (unused for primitives)
    atom_id = hilbert_id(byte_value, 0, 0, 0)
    
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO atoms (id, geom)
            VALUES (%s, ST_GeomFromEWKT('POINT ZM(%s 0 0 0)'))
            ON CONFLICT (id) DO NOTHING
        """, (atom_id, byte_value))
    
    return atom_id

def ingest_composite_atom(conn, child_ids, modality):
    """Create composite atom from child atoms using LINESTRING with M-coordinates"""
    if not child_ids:
        return None
    
    # Build LINESTRING where each point's M-coordinate references a child atom
    points = []
    for idx, child_id in enumerate(child_ids):
        # X = sequence position
        # Y = modality type
        # Z = 0
        # M = reference to child atom ID
        points.append(f"{idx} {modality} 0 {child_id}")
    
    linestring_wkt = f"LINESTRING ZM({', '.join(points)})"
    
    # Compute deterministic ID from the composition
    composition_hash = hashlib.sha256(str(child_ids).encode()).digest()
    hash_int = int.from_bytes(composition_hash[:8], 'big')
    
    # Use hash to generate 4D coords for Hilbert
    x = hash_int & 0xFFFF
    y = (hash_int >> 16) & 0xFFFF
    z = (hash_int >> 32) & 0xFFFF
    m = modality
    
    atom_id = hilbert_id(x, y, z, m)
    
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO atoms (id, geom)
            VALUES (%s, ST_GeomFromEWKT(%s))
            ON CONFLICT (id) DO NOTHING
        """, (atom_id, linestring_wkt))
    
    return atom_id

def ingest_string(conn, text):
    """Decompose string to character atoms, then build composite"""
    # Ingest each character as byte atoms
    char_ids = []
    for char in text:
        byte_val = ord(char)
        char_id = ingest_byte_atom(conn, byte_val)
        char_ids.append(char_id)
    
    # Create composite atom for the string (modality=1 for text)
    string_id = ingest_composite_atom(conn, char_ids, 1)
    
    return string_id

def ingest_number(conn, value):
    """Ingest numeric value as atom"""
    # Convert float to int representation for Hilbert
    if isinstance(value, float):
        # Use IEEE 754 representation
        bytes_val = struct.pack('!f', value)
        int_val = int.from_bytes(bytes_val, 'big')
    else:
        int_val = int(value)
    
    # X = integer representation
    # Y = modality (2 = number)
    atom_id = hilbert_id(int_val & 0xFFFF, 2, (int_val >> 16) & 0xFFFF, (int_val >> 32) & 0xFFFF)
    
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO atoms (id, geom)
            VALUES (%s, ST_GeomFromEWKT('POINT ZM(%s 2 %s %s)'))
            ON CONFLICT (id) DO NOTHING
        """, (atom_id, int_val & 0xFFFF, (int_val >> 16) & 0xFFFF, (int_val >> 32) & 0xFFFF))
    
    return atom_id

def ingest_tokenizer(conn, tokenizer_path):
    """Ingest tokenizer vocabulary"""
    print(f"Loading tokenizer from {tokenizer_path}")
    
    with open(tokenizer_path) as f:
        tokenizer_data = json.load(f)
    
    vocab = tokenizer_data['model']['vocab']
    print(f"Found {len(vocab)} tokens")
    
    token_ids = {}
    for token_str, token_idx in vocab.items():
        # Decompose token to bytes
        token_bytes = token_str.encode('utf-8', errors='replace')
        
        # Ingest each byte
        byte_ids = []
        for byte_val in token_bytes:
            byte_id = ingest_byte_atom(conn, byte_val)
            byte_ids.append(byte_id)
        
        # Create composite for token (modality=3)
        token_id = ingest_composite_atom(conn, byte_ids, 3)
        token_ids[token_idx] = token_id
    
    conn.commit()
    print(f"Ingested {len(token_ids)} tokens")
    return token_ids

def ingest_weights(conn, weights_file, token_ids):
    """Ingest weight tensors as relations between tokens"""
    print(f"Loading weights from {weights_file}")
    
    # Load GGUF file (simplified - just get first tensor for demo)
    with open(weights_file, 'rb') as f:
        # Skip GGUF header (this is simplified)
        # In production, use proper GGUF parser
        pass
    
    print("Weight ingestion placeholder - need proper GGUF parser")
    # TODO: Parse GGUF format properly and create weight relation atoms
    
    return

def main():
    # Connect to database
    conn = psycopg2.connect(
        dbname='hartonomous',
        user='postgres',
        host='localhost'
    )
    conn.autocommit = False
    
    # Create index
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX IF NOT EXISTS atoms_geom_idx ON atoms USING GIST(geom);")
    conn.commit()
    
    # Ingest tokenizer
    tokenizer_path = '/data/models/qwen2.5-0.5b-instruct-q8_0.gguf.tokenizer.json'
    token_ids = ingest_tokenizer(conn, tokenizer_path)
    
    # Check results
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT ST_GeometryType(geom)) FROM atoms")
        total, geom_types = cur.fetchone()
        print(f"\nTotal atoms: {total:,}")
        print(f"Geometry types: {geom_types}")
    
    conn.close()
    print("\n✓ Ingestion complete")

if __name__ == '__main__':
    main()
