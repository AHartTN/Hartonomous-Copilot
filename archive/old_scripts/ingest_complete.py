#!/usr/bin/env python3
"""Complete unified model ingestion."""
import psycopg2
import numpy as np
from safetensors.torch import load_file
import json
import hashlib
from tqdm import tqdm
from io import StringIO

DB_PARAMS = {'dbname': 'hartonomous', 'user': 'postgres', 'host': 'localhost'}

def get_hilbert_coord(value_bytes, modality):
    """Generate deterministic 4D coordinate."""
    h = hashlib.sha256(value_bytes + modality.encode()).digest()
    coords = np.frombuffer(h[:32], dtype=np.float64)
    coords = (coords / np.max(np.abs(coords))) if np.max(np.abs(coords)) > 0 else coords
    return coords[:4]

def main():
    print("=== UNIFIED INGESTION ===\n")
    
    tensors = load_file("/data/models/qwen2.5-0.5b/model.safetensors")
    with open("/data/models/qwen2.5-0.5b/tokenizer.json") as f:
        tokenizer = json.load(f)
    vocab = tokenizer['model']['vocab']
    
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = False
    cur = conn.cursor()
    
    # Clear existing data
    cur.execute("TRUNCATE atoms, relations CASCADE")
    conn.commit()
    
    atom_registry = {}
    next_id = 1
    
    def create_atom(value, modality):
        nonlocal next_id
        if isinstance(value, (int, float, np.number)):
            vb = np.array([value], dtype=np.float32).tobytes()
        elif isinstance(value, str):
            vb = value.encode('utf-8')
        else:
            vb = str(value).encode('utf-8')
        
        ch = hashlib.sha256(vb).hexdigest()
        if ch in atom_registry:
            return atom_registry[ch]
        
        coords = get_hilbert_coord(vb, modality)
        aid = next_id
        next_id += 1
        geom = f"POINT Z ({coords[0]} {coords[1]} {coords[2]})"
        atom_registry[ch] = (aid, geom, vb)
        return aid
    
    print("[1/3] Vocabulary...")
    for token_str in tqdm(list(vocab.keys())[:5000], desc="Tokens"):
        create_atom(token_str, 'token')
    
    # Insert atoms
    buf = StringIO()
    for aid, geom, raw in atom_registry.values():
        buf.write(f"{aid}\t{geom}\t\\\\x{raw.hex()}\n")
    buf.seek(0)
    cur.copy_from(buf, 'atoms', columns=['id', 'geometry', 'raw_value'])
    conn.commit()
    print(f"✓ {len(atom_registry)} atoms")
    
    # Build token lookup
    token_map = {}
    for ts in vocab.keys():
        ch = hashlib.sha256(ts.encode()).hexdigest()
        if ch in atom_registry:
            token_map[ts] = atom_registry[ch][0]
    
    print(f"\n[2/3] Embeddings...")
    embeddings = tensors['model.embed_tokens.weight'].numpy()
    
    rel_buf = StringIO()
    count = 0
    for token_str, token_id in tqdm(list(vocab.items())[:1000], desc="Embed"):
        if token_str not in token_map:
            continue
        ta = token_map[token_str]
        for dim, val in enumerate(embeddings[token_id][:100]):  # First 100 dims
            wa = create_atom(float(val), 'weight')
            rel_buf.write(f"{ta}\t{wa}\tembedding\t\\N\t{dim}\t{{}}\n")
            count += 1
    
    # Insert new atoms
    new_atoms = StringIO()
    for aid, geom, raw in atom_registry.values():
        new_atoms.write(f"{aid}\t{geom}\t\\\\x{raw.hex()}\n")
    new_atoms.seek(0)
    cur.execute("TRUNCATE atoms CASCADE")
    cur.copy_from(new_atoms, 'atoms', columns=['id', 'geometry', 'raw_value'])
    
    rel_buf.seek(0)
    cur.copy_from(rel_buf, 'relations', columns=['parent_id', 'child_id', 'relation_type', 'weight_atom_id', 'position', 'meta'])
    conn.commit()
    print(f"✓ {count} embedding relations")
    
    print(f"\n[3/3] Layer 0 attention...")
    qw = tensors['model.layers.0.self_attn.q_proj.weight'].numpy()
    print(f"Q matrix: {qw.shape}")
    
    # Sample top 0.1% weights
    flat = qw.flatten()
    thresh = np.percentile(np.abs(flat), 99.9)
    
    rel_buf2 = StringIO()
    count2 = 0
    for idx, val in enumerate(tqdm(flat, desc="Q weights")):
        if abs(val) < thresh:
            continue
        row, col = divmod(idx, qw.shape[1])
        ia = create_atom(col, 'neuron')
        oa = create_atom(row, 'neuron')
        wa = create_atom(float(val), 'weight')
        rel_buf2.write(f"{ia}\t{oa}\tattn_q\t{wa}\t\\N\t{{\"r\":{row},\"c\":{col}}}\n")
        count2 += 1
    
    # Insert atoms again
    final_atoms = StringIO()
    for aid, geom, raw in atom_registry.values():
        final_atoms.write(f"{aid}\t{geom}\t\\\\x{raw.hex()}\n")
    final_atoms.seek(0)
    cur.execute("TRUNCATE atoms CASCADE")
    cur.copy_from(final_atoms, 'atoms', columns=['id', 'geometry', 'raw_value'])
    
    rel_buf2.seek(0)
    cur.copy_from(rel_buf2, 'relations', columns=['parent_id', 'child_id', 'relation_type', 'weight_atom_id', 'position', 'meta'])
    conn.commit()
    
    cur.execute("SELECT COUNT(*) FROM atoms")
    ac = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM relations")
    rc = cur.fetchone()[0]
    
    print(f"\n✓ COMPLETE: {ac:,} atoms, {rc:,} relations")
    
    cur.close()
    conn.close()

if __name__ == '__main__':
    main()
