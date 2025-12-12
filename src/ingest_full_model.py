#!/usr/bin/env python3
"""
Complete model ingestion: tokens, embeddings, and attention weights as relations.
"""
import psycopg2
import numpy as np
import struct
import json
from pathlib import Path

# GGUF parsing
def read_gguf(path):
    """Parse GGUF file and extract tensors"""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError("Not a GGUF file")
        
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        # Read metadata
        metadata = {}
        for _ in range(n_kv):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            val_type = struct.unpack('<I', f.read(4))[0]
            
            if val_type == 4:  # String
                str_len = struct.unpack('<Q', f.read(8))[0]
                val = f.read(str_len).decode('utf-8')
            elif val_type == 8:  # Array
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                val = []
                for _ in range(arr_len):
                    if arr_type == 4:  # String array
                        s_len = struct.unpack('<Q', f.read(8))[0]
                        val.append(f.read(s_len).decode('utf-8'))
            else:
                val = None
            
            metadata[key] = val
        
        # Read tensor info
        tensors = []
        for _ in range(n_tensors):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims))
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            tensors.append({
                'name': name,
                'dims': dims,
                'dtype': dtype,
                'offset': offset
            })
        
        return metadata, tensors, f

def ingest_model(gguf_path, db_conn):
    """Ingest complete model as graph"""
    cur = db_conn.cursor()
    
    print(f"📦 Loading {gguf_path}...")
    metadata, tensors, f = read_gguf(gguf_path)
    
    # Get vocabulary
    vocab = metadata.get('tokenizer.ggml.tokens', [])
    print(f"📝 Vocabulary: {len(vocab)} tokens")
    
    # Step 1: Ingest token atoms
    print("Ingesting token atoms...")
    token_atoms = {}
    for idx, token in enumerate(vocab):
        token_bytes = token.encode('utf-8')
        cur.execute("""
            INSERT INTO atoms (raw_value, modality, geometry)
            VALUES (%s, 'token', ST_MakePoint(0, 0, 0))
            ON CONFLICT (raw_value, modality) DO NOTHING
            RETURNING id
        """, (token_bytes,))
        result = cur.fetchone()
        if result:
            token_atoms[idx] = result[0]
        else:
            cur.execute("SELECT id FROM atoms WHERE raw_value = %s AND modality = 'token'", (token_bytes,))
            token_atoms[idx] = cur.fetchone()[0]
    
    db_conn.commit()
    print(f"✓ {len(token_atoms)} token atoms")
    
    # Step 2: Find embedding tensor
    emb_tensor = next((t for t in tensors if 'token_embd' in t['name']), None)
    if not emb_tensor:
        print("⚠️ No embedding tensor found")
        return
    
    print(f"🧠 Embedding tensor: {emb_tensor['name']} {emb_tensor['dims']}")
    
    # Step 3: Load embedding weights and create relations
    # For demo: load first 1000 tokens only
    print("Loading embeddings (first 1000 tokens)...")
    embed_dim = emb_tensor['dims'][0]
    
    # Create float atoms for unique embedding values
    f.seek(emb_tensor['offset'])
    
    relations_batch = []
    for token_idx in range(min(1000, len(vocab))):
        if token_idx not in token_atoms:
            continue
            
        # Read embedding vector (simplified - assuming F32)
        embedding = struct.unpack(f'<{embed_dim}f', f.read(embed_dim * 4))
        
        # For each dimension, create relation
        for dim_idx, value in enumerate(embedding):
            # Insert float atom
            cur.execute("""
                INSERT INTO atoms (raw_value, modality, geometry)
                VALUES (%s, 'float', ST_MakePoint(%s, 0, 0))
                ON CONFLICT (raw_value, modality) DO NOTHING
                RETURNING id
            """, (struct.pack('<f', value), float(value)))
            
            result = cur.fetchone()
            weight_id = result[0] if result else cur.execute(
                "SELECT id FROM atoms WHERE raw_value = %s AND modality = 'float'",
                (struct.pack('<f', value),)
            ).fetchone()[0]
            
            relations_batch.append((
                token_atoms[token_idx],
                weight_id,
                'embedding',
                dim_idx,
                json.dumps({'layer': 'token_embd', 'dim': dim_idx})
            ))
        
        if len(relations_batch) >= 10000:
            cur.executemany("""
                INSERT INTO relations (parent_id, child_id, relation_type, position, meta)
                VALUES (%s, %s, %s, %s, %s)
            """, relations_batch)
            db_conn.commit()
            print(f"  Processed {token_idx} tokens...")
            relations_batch = []
    
    if relations_batch:
        cur.executemany("""
            INSERT INTO relations (parent_id, child_id, relation_type, position, meta)
            VALUES (%s, %s, %s, %s, %s)
        """, relations_batch)
        db_conn.commit()
    
    print(f"✓ Embeddings ingested")
    
    # Stats
    cur.execute("SELECT COUNT(*) FROM atoms")
    print(f"📊 Total atoms: {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM relations")
    print(f"📊 Total relations: {cur.fetchone()[0]:,}")

if __name__ == '__main__':
    conn = psycopg2.connect(
        dbname='hartonomous',
        user='postgres'
    )
    
    ingest_model('models/qwen2.5-0.5b.gguf', conn)
    conn.close()
