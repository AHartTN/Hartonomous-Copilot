#!/usr/bin/env python3
"""
HARTONOMOUS - Complete Atomic Ingestion System
The Periodic Table of AI
"""
import psycopg2
import numpy as np
import json
import struct
from pathlib import Path
import hashlib

# Hilbert curve implementation
def hilbert_encode_4d(x, y, z, w, order=8):  # Reduced from 16 to 8 to fit in 64-bit
    """Encode 4D point to Hilbert curve index"""
    def interleave_bits(coords, order):
        result = 0
        for i in range(order):
            for coord in coords:
                bit = (coord >> i) & 1
                result = (result << 1) | bit
        return result
    
    # Normalize [-1,1] to [0, 2^order-1]
    max_val = (1 << order) - 1
    xi = max(0, min(max_val, int((x + 1) * max_val / 2)))
    yi = max(0, min(max_val, int((y + 1) * max_val / 2)))
    zi = max(0, min(max_val, int((z + 1) * max_val / 2)))
    wi = max(0, min(max_val, int((w + 1) * max_val / 2)))
    
    return interleave_bits([xi, yi, zi, wi], order)

class AtomicSystem:
    def __init__(self, conn):
        self.conn = conn
        self.cur = conn.cursor()
        self.atom_cache = {}  # content_hash -> (id, coords)
        self.landmarks = None
        
    def seed_landmarks(self):
        """Create fundamental constant atoms - the periodic table"""
        print("🌱 Seeding fundamental constants...")
        
        # Seed with basic building blocks
        primitives = []
        
        # Numbers 0-9
        for i in range(10):
            primitives.append((f'num_{i}', str(i).encode()))
        
        # ASCII printable characters (32-126)
        for i in range(32, 127):
            char = chr(i)
            primitives.append((f'char_{i}', char.encode()))
        
        # Common whitespace
        primitives.append(('space', b' '))
        primitives.append(('newline', b'\n'))
        primitives.append(('tab', b'\t'))
        
        print(f"   Creating {len(primitives)} primitive atoms...")
        
        # Project to 4D hypercube corners using deterministic method
        np.random.seed(42)  # Deterministic
        
        for idx, (name, content) in enumerate(primitives):
            # Deterministic coordinates from content hash
            h = hashlib.sha256(content).digest()
            coords = []
            for i in range(4):
                # Use different bytes for each dimension
                byte_val = struct.unpack('<I', h[i*4:(i+1)*4])[0]
                # Normalize unsigned int to [-1, 1]
                normalized = (byte_val / (2**32 - 1)) * 2 - 1
                coords.append(normalized)
            
            x, y, z, w = coords
            hilbert_id = hilbert_encode_4d(x, y, z, w)
            
            self.cur.execute(
                "INSERT INTO atoms (id, geom, raw_value) VALUES (%s, ST_MakePoint(%s, %s, %s, %s), %s) ON CONFLICT DO NOTHING",
                (int(hilbert_id), float(x), float(y), float(z), float(w), content)
            )
            self.atom_cache[content] = (hilbert_id, coords)
        
        self.conn.commit()
        print(f"   ✓ {len(primitives)} primitive atoms seeded")
        
        # Store landmarks for interpolation
        self.landmarks = {content: coords for content, (_, coords) in self.atom_cache.items()}
    
    def get_or_create_atom(self, content):
        """Get existing atom or create new one via geometric interpolation"""
        if content in self.atom_cache:
            return self.atom_cache[content]
        
        # Decompose content into sub-atoms
        if len(content) == 1:
            # Single character - hash to coordinates
            h = hashlib.sha256(content).digest()
            coords = []
            for i in range(4):
                byte_val = struct.unpack('<I', h[i*4:(i+1)*4])[0]
                normalized = (byte_val / (2**32 - 1)) * 2 - 1
                coords.append(normalized)
        else:
            # Composite - average of constituent atoms
            sub_atoms = []
            for byte in content:
                sub_content = bytes([byte])
                sub_id, sub_coords = self.get_or_create_atom(sub_content)
                sub_atoms.append(sub_coords)
            
            # Centroid of sub-atoms
            coords = np.mean(sub_atoms, axis=0).tolist()
        
        x, y, z, w = coords
        hilbert_id = hilbert_encode_4d(x, y, z, w)
        
        self.cur.execute(
            "INSERT INTO atoms (id, geom, raw_value) VALUES (%s, ST_MakePoint(%s, %s, %s, %s), %s) ON CONFLICT DO NOTHING",
            (int(hilbert_id), float(x), float(y), float(z), float(w), content)
        )
        
        self.atom_cache[content] = (hilbert_id, coords)
        return hilbert_id, coords
    
    def ingest_tokenizer(self, tokenizer_path):
        """Ingest vocabulary with proper byte-level decomposition"""
        print(f"\n📚 Ingesting tokenizer: {tokenizer_path}")
        
        with open(tokenizer_path) as f:
            tokenizer = json.load(f)
        
        vocab = tokenizer['model']['vocab']
        print(f"   Processing {len(vocab)} tokens...")
        
        token_atoms = []
        batch_size = 1000
        
        for token_str, token_id in vocab.items():
            # Decode BPE token to actual bytes
            token_bytes = token_str.encode('utf-8')
            
            # Get atom for full token
            atom_id, coords = self.get_or_create_atom(token_bytes)
            token_atoms.append((token_id, atom_id))
            
            # Store composition (token -> constituent bytes)
            for pos, byte_val in enumerate(token_bytes):
                byte_atom_id, _ = self.get_or_create_atom(bytes([byte_val]))
                self.cur.execute(
                    "INSERT INTO compositions (parent_id, child_id, position) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (atom_id, byte_atom_id, pos)
                )
            
            if len(token_atoms) % batch_size == 0:
                self.conn.commit()
                print(f"   ✓ {len(token_atoms)} tokens processed...")
        
        self.conn.commit()
        print(f"   ✓ {len(token_atoms)} tokens ingested")
        return dict(token_atoms)
    
    def ingest_model_weights(self, model_path, token_map):
        """Ingest model tensors as weight connections"""
        print(f"\n🧠 Ingesting model weights: {model_path}")
        
        import torch
        from safetensors import safe_open
        
        with safe_open(model_path, framework="pt") as f:
            tensor_names = f.keys()
            print(f"   Found {len(list(tensor_names))} tensors")
            
            for name in f.keys():
                tensor = f.get_tensor(name)
                print(f"   Processing {name}: {tensor.shape}")
                
                # Only ingest embedding and small attention layers for demo
                if 'embed' in name or ('attn' in name and tensor.numel() < 100000):
                    self._ingest_tensor(name, tensor, token_map)
                    self.conn.commit()
        
        print("   ✓ Model weights ingested")
    
    def _ingest_tensor(self, layer_name, tensor, token_map):
        """Ingest individual tensor as atomic connections"""
        data = tensor.cpu().numpy()
        
        if 'embed' in layer_name and len(data.shape) == 2:
            # Embedding matrix: vocab_size x embed_dim
            # Each row is a token's embedding vector
            print(f"      Embedding: {data.shape[0]} tokens x {data.shape[1]} dims")
            
            # Sample subset for demo
            for token_id in range(min(100, data.shape[0])):
                if token_id not in token_map:
                    continue
                    
                token_atom = token_map[token_id]
                
                # Store embedding as connections to dimension atoms
                for dim_idx in range(min(10, data.shape[1])):  # Sample first 10 dims
                    weight_val = float(data[token_id, dim_idx])
                    weight_bytes = struct.pack('f', weight_val)
                    weight_atom_id, _ = self.get_or_create_atom(weight_bytes)
                    
                    # Create dimension atom
                    dim_atom_id, _ = self.get_or_create_atom(f"dim_{dim_idx}".encode())
                    
                    # Connection: token --[weight]--> dimension
                    self.cur.execute("""
                        INSERT INTO connections (from_id, to_id, weight_id, layer)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (token_atom, dim_atom_id, weight_atom_id, layer_name))

def main():
    # Connect to database
    conn = psycopg2.connect("dbname=hartonomous user=postgres")
    
    system = AtomicSystem(conn)
    
    # 1. Seed fundamental constants
    system.seed_landmarks()
    
    # Model path from Hugging Face
    model_dir = "/tmp/hf_models/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    
    # 2. Ingest vocabulary
    token_map = system.ingest_tokenizer(f'{model_dir}/tokenizer.json')
    
    # 3. Ingest model weights
    system.ingest_model_weights(f'{model_dir}/model.safetensors', token_map)
    
    # Statistics
    system.cur.execute("SELECT COUNT(*) FROM atoms")
    atom_count = system.cur.fetchone()[0]
    
    system.cur.execute("SELECT COUNT(*) FROM compositions")
    comp_count = system.cur.fetchone()[0]
    
    system.cur.execute("SELECT COUNT(*) FROM connections")
    conn_count = system.cur.fetchone()[0]
    
    print(f"\n{'='*60}")
    print(f"HARTONOMOUS SYSTEM READY")
    print(f"{'='*60}")
    print(f"Atoms:        {atom_count:,}")
    print(f"Compositions: {comp_count:,}")
    print(f"Connections:  {conn_count:,}")
    print(f"{'='*60}\n")
    
    conn.close()

if __name__ == '__main__':
    main()
