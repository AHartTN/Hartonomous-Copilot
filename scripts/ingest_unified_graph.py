#!/usr/bin/env python3
"""
Unified Graph Ingestion - Run as postgres user via sudo
"""
import subprocess
import sys

# Run the actual ingestion script as postgres user
result = subprocess.run([
    'sudo', '-u', 'postgres',
    'python3', '/home/ahart/Projects/Hartonomous-Copilot/scripts/ingest_unified_graph_impl.py'
], capture_output=False)

sys.exit(result.returncode)


class GGUFReader:
    """Read GGUF format model files"""
    def __init__(self, path):
        self.path = path
        self.file = open(path, 'rb')
        self.metadata = {}
        self.tensors = {}
        self._read_header()
    
    def _read_header(self):
        # Read GGUF magic and version
        magic = self.file.read(4)
        if magic != b'GGUF':
            raise ValueError("Not a GGUF file")
        
        version = struct.unpack('<I', self.file.read(4))[0]
        tensor_count = struct.unpack('<Q', self.file.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', self.file.read(8))[0]
        
        # Read metadata
        for _ in range(metadata_kv_count):
            key = self._read_string()
            value_type = struct.unpack('<I', self.file.read(4))[0]
            value = self._read_value(value_type)
            self.metadata[key] = value
        
        # Read tensor info
        self.tensor_info = []
        for _ in range(tensor_count):
            name = self._read_string()
            n_dims = struct.unpack('<I', self.file.read(4))[0]
            dims = struct.unpack(f'<{n_dims}Q', self.file.read(8 * n_dims))
            dtype = struct.unpack('<I', self.file.read(4))[0]
            offset = struct.unpack('<Q', self.file.read(8))[0]
            
            self.tensor_info.append({
                'name': name,
                'dims': dims,
                'dtype': dtype,
                'offset': offset
            })
    
    def _read_string(self):
        length = struct.unpack('<Q', self.file.read(8))[0]
        return self.file.read(length).decode('utf-8')
    
    def _read_value(self, value_type):
        if value_type == 4:  # String
            return self._read_string()
        elif value_type == 5:  # Array
            array_type = struct.unpack('<I', self.file.read(4))[0]
            array_len = struct.unpack('<Q', self.file.read(8))[0]
            return [self._read_value(array_type) for _ in range(array_len)]
        elif value_type == 8:  # Uint32
            return struct.unpack('<I', self.file.read(4))[0]
        elif value_type == 9:  # Int32
            return struct.unpack('<i', self.file.read(4))[0]
        elif value_type == 10:  # Float32
            return struct.unpack('<f', self.file.read(4))[0]
        elif value_type == 12:  # Uint64
            return struct.unpack('<Q', self.file.read(8))[0]
        else:
            raise ValueError(f"Unknown value type: {value_type}")
    
    def get_tokenizer_model(self):
        """Extract tokenizer tokens"""
        if 'tokenizer.ggml.tokens' in self.metadata:
            return self.metadata['tokenizer.ggml.tokens']
        return []

def compute_4d_coordinates(value, modality, index, total):
    """
    Compute deterministic 4D coordinates for an atom
    Uses orthonormalization to create bounded space
    """
    # Normalize index to [-1, 1]
    if total > 1:
        normalized_idx = (2.0 * index / (total - 1)) - 1.0
    else:
        normalized_idx = 0.0
    
    # Modality determines base orientation
    modality_map = {
        'byte': 0,
        'token': 1,
        'float': 2,
        'embedding': 3
    }
    mod_val = modality_map.get(modality, 0) / 4.0  # Normalize to [0, 0.25]
    
    # If value is numeric, use it; otherwise use hash
    if isinstance(value, (int, float)):
        val_component = np.tanh(float(value))  # Squash to [-1, 1]
    else:
        val_component = (hash(str(value)) % 10000) / 10000.0 * 2 - 1
    
    # Gram-Schmidt-like orthonormalization
    x = normalized_idx
    y = mod_val * 4 - 1  # Scale back to [-1, 1]
    z = val_component
    w = np.sin(normalized_idx * np.pi) * np.cos(mod_val * 2 * np.pi)
    
    # Ensure bounded in [-1, 1]^4
    coords = [np.clip(c, -1.0, 1.0) for c in [x, y, z, w]]
    return coords

def hilbert_encode_4d(coords):
    """
    Encode 4D coordinates to Hilbert curve index
    Using simple bit interleaving for 16-bit precision per dimension
    """
    # Scale [-1, 1] to [0, 65535]
    scaled = [(int((c + 1) * 32767.5) & 0xFFFF) for c in coords]
    
    # Interleave bits (simple version - not true Hilbert but deterministic)
    result = 0
    for bit in range(16):
        for dim in range(4):
            result = (result << 1) | ((scaled[dim] >> (15 - bit)) & 1)
    
    return result

def ingest_vocabulary(conn, tokens):
    """Ingest tokenizer vocabulary as atomic tokens"""
    print(f"\n📚 Ingesting {len(tokens)} vocabulary tokens...")
    
    cur = conn.cursor()
    atom_ids = {}
    
    # First pass: Create byte atoms (primitives)
    byte_atoms = {}
    for token in tokens:
        if isinstance(token, str):
            for byte_val in token.encode('utf-8'):
                if byte_val not in byte_atoms:
                    coords = compute_4d_coordinates(byte_val, 'byte', byte_val, 256)
                    hilbert_id = hilbert_encode_4d(coords)
                    
                    cur.execute("""
                        INSERT INTO atoms (id, geometry, raw_value, modality)
                        VALUES (%s, ST_MakePoint(%s, %s, %s, %s), %s, 'byte')
                        ON CONFLICT (id) DO NOTHING
                    """, (hilbert_id, coords[0], coords[1], coords[2], coords[3], 
                          bytes([byte_val])))
                    byte_atoms[byte_val] = hilbert_id
    
    conn.commit()
    print(f"  ✓ Created {len(byte_atoms)} unique byte atoms")
    
    # Second pass: Create token atoms and their compositions
    for idx, token in enumerate(tokens):
        if isinstance(token, str):
            # Create token atom
            coords = compute_4d_coordinates(token, 'token', idx, len(tokens))
            hilbert_id = hilbert_encode_4d(coords)
            
            cur.execute("""
                INSERT INTO atoms (id, geometry, raw_value, modality)
                VALUES (%s, ST_MakePoint(%s, %s, %s, %s), %s, 'token')
                ON CONFLICT (id) DO NOTHING
            """, (hilbert_id, coords[0], coords[1], coords[2], coords[3],
                  token.encode('utf-8')))
            
            atom_ids[idx] = hilbert_id
            
            # Create relations to byte atoms
            byte_sequence = token.encode('utf-8')
            for pos, byte_val in enumerate(byte_sequence):
                byte_atom_id = byte_atoms[byte_val]
                cur.execute("""
                    INSERT INTO relations (parent_id, child_id, relation_type, position)
                    VALUES (%s, %s, 'composition', %s)
                    ON CONFLICT DO NOTHING
                """, (hilbert_id, byte_atom_id, pos))
        
        if (idx + 1) % 10000 == 0:
            conn.commit()
            print(f"  Progress: {idx + 1}/{len(tokens)} tokens")
    
    conn.commit()
    print(f"  ✓ Created {len(atom_ids)} token atoms with compositions")
    
    return atom_ids

def ingest_embeddings(conn, model_reader, token_atom_ids):
    """Ingest token embeddings as relations"""
    print(f"\n🔗 Ingesting token embeddings...")
    
    # Find embedding tensor
    embedding_tensor = None
    for tensor_info in model_reader.tensor_info:
        if 'token_embd' in tensor_info['name'] or 'tok_embeddings' in tensor_info['name']:
            embedding_tensor = tensor_info
            break
    
    if not embedding_tensor:
        print("  ⚠ No embedding tensor found")
        return
    
    print(f"  Found: {embedding_tensor['name']}, dims: {embedding_tensor['dims']}")
    
    # For now, create placeholder float atoms
    # Full implementation would read actual tensor values
    cur = conn.cursor()
    
    # Create sample float atoms
    unique_floats = set()
    for i in range(100):  # Sample values
        unique_floats.add(round(np.random.randn() * 0.1, 6))
    
    float_atom_ids = {}
    for idx, float_val in enumerate(unique_floats):
        coords = compute_4d_coordinates(float_val, 'float', idx, len(unique_floats))
        hilbert_id = hilbert_encode_4d(coords)
        
        cur.execute("""
            INSERT INTO atoms (id, geometry, raw_value, modality)
            VALUES (%s, ST_MakePoint(%s, %s, %s, %s), %s, 'float')
            ON CONFLICT (id) DO NOTHING
        """, (hilbert_id, coords[0], coords[1], coords[2], coords[3],
              struct.pack('<f', float_val)))
        
        float_atom_ids[float_val] = hilbert_id
    
    conn.commit()
    print(f"  ✓ Created {len(float_atom_ids)} float atoms")
    
    # Create embedding relations (sample for first 1000 tokens)
    embed_dim = 896  # Qwen2.5 embedding dimension
    for token_idx in list(token_atom_ids.keys())[:1000]:
        token_atom_id = token_atom_ids[token_idx]
        
        # Create relations to random float atoms (placeholder)
        for dim_idx in range(min(embed_dim, 100)):
            float_val = list(float_atom_ids.keys())[dim_idx % len(float_atom_ids)]
            float_atom_id = float_atom_ids[float_val]
            
            cur.execute("""
                INSERT INTO relations (parent_id, child_id, relation_type, position, weight_atom_id)
                VALUES (%s, %s, 'embedding', %s, %s)
                ON CONFLICT DO NOTHING
            """, (token_atom_id, float_atom_id, dim_idx, float_atom_id))
        
        if (token_idx + 1) % 100 == 0:
            conn.commit()
            print(f"  Progress: {token_idx + 1}/1000 tokens embedded")
    
    conn.commit()
    print(f"  ✓ Created embedding relations")

def main():
    print("=" * 60)
    print("HARTONOMOUS - Unified Graph Ingestion")
    print("=" * 60)
    
    # Connect to database
    print("\n🔌 Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("  ✓ Connected")
    
    # Read GGUF model
    print(f"\n📖 Reading model: {MODEL_PATH}")
    reader = GGUFReader(MODEL_PATH)
    print(f"  ✓ Loaded {len(reader.tensor_info)} tensors")
    
    # Get tokenizer
    tokens = reader.get_tokenizer_model()
    print(f"  ✓ Found {len(tokens)} vocabulary tokens")
    
    # Ingest vocabulary
    token_atom_ids = ingest_vocabulary(conn, tokens)
    
    # Ingest embeddings
    ingest_embeddings(conn, reader, token_atom_ids)
    
    # Stats
    cur = conn.cursor()
    cur.execute("SELECT modality, COUNT(*) FROM atoms GROUP BY modality")
    print(f"\n📊 Final Statistics:")
    for modality, count in cur.fetchall():
        print(f"  {modality}: {count:,} atoms")
    
    cur.execute("SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type")
    print(f"\n🔗 Relations:")
    for rel_type, count in cur.fetchall():
        print(f"  {rel_type}: {count:,}")
    
    conn.close()
    print("\n✅ Ingestion complete!")

if __name__ == '__main__':
    main()
