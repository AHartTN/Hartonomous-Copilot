#!/usr/bin/env python3
"""
Unified Graph Ingestion Implementation
Reads GGUF model and creates atomic graph structure
"""
import struct
import numpy as np
import psycopg2
from pathlib import Path

MODEL_PATH = Path('/data/models/qwen2.5-0.5b-instruct.Q8_0.gguf')

class GGUFReader:
    """Read GGUF format model files"""
    def __init__(self, path):
        self.path = path
        self.file = open(path, 'rb')
        self.metadata = {}
        self.tensors = {}
        self._read_header()
    
    def _read_header(self):
        magic = self.file.read(4)
        if magic != b'GGUF':
            raise ValueError("Not a GGUF file")
        
        version = struct.unpack('<I', self.file.read(4))[0]
        tensor_count = struct.unpack('<Q', self.file.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', self.file.read(8))[0]
        
        for _ in range(metadata_kv_count):
            key = self._read_string()
            value_type = struct.unpack('<I', self.file.read(4))[0]
            value = self._read_value(value_type)
            self.metadata[key] = value
        
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
        if value_type == 4:
            return self._read_string()
        elif value_type == 5:
            array_type = struct.unpack('<I', self.file.read(4))[0]
            array_len = struct.unpack('<Q', self.file.read(8))[0]
            return [self._read_value(array_type) for _ in range(array_len)]
        elif value_type == 8:
            return struct.unpack('<I', self.file.read(4))[0]
        elif value_type == 9:
            return struct.unpack('<i', self.file.read(4))[0]
        elif value_type == 10:
            return struct.unpack('<f', self.file.read(4))[0]
        elif value_type == 12:
            return struct.unpack('<Q', self.file.read(8))[0]
        else:
            raise ValueError(f"Unknown value type: {value_type}")
    
    def get_tokenizer_model(self):
        if 'tokenizer.ggml.tokens' in self.metadata:
            return self.metadata['tokenizer.ggml.tokens']
        return []

def compute_4d_coordinates(value, modality, index, total):
    if total > 1:
        normalized_idx = (2.0 * index / (total - 1)) - 1.0
    else:
        normalized_idx = 0.0
    
    modality_map = {'byte': 0, 'token': 1, 'float': 2, 'embedding': 3}
    mod_val = modality_map.get(modality, 0) / 4.0
    
    if isinstance(value, (int, float)):
        val_component = np.tanh(float(value))
    else:
        val_component = (hash(str(value)) % 10000) / 10000.0 * 2 - 1
    
    x = normalized_idx
    y = mod_val * 4 - 1
    z = val_component
    w = np.sin(normalized_idx * np.pi) * np.cos(mod_val * 2 * np.pi)
    
    coords = [np.clip(c, -1.0, 1.0) for c in [x, y, z, w]]
    return coords

def hilbert_encode_4d(coords):
    scaled = [(int((c + 1) * 32767.5) & 0xFFFF) for c in coords]
    result = 0
    for bit in range(16):
        for dim in range(4):
            result = (result << 1) | ((scaled[dim] >> (15 - bit)) & 1)
    return result

def main():
    print("=" * 60)
    print("HARTONOMOUS - Unified Graph Ingestion")
    print("=" * 60)
    
    conn = psycopg2.connect(dbname='hartonomous')
    print("\n🔌 Connected to PostgreSQL")
    
    reader = GGUFReader(MODEL_PATH)
    tokens = reader.get_tokenizer_model()
    print(f"\n📖 Found {len(tokens)} tokens, {len(reader.tensor_info)} tensors")
    
    cur = conn.cursor()
    
    # Ingest bytes
    print("\n📚 Ingesting byte atoms...")
    byte_atoms = {}
    all_bytes = set()
    for token in tokens[:10000]:
        if isinstance(token, str):
            all_bytes.update(token.encode('utf-8'))
    
    for byte_val in sorted(all_bytes):
        coords = compute_4d_coordinates(byte_val, 'byte', byte_val, 256)
        hilbert_id = hilbert_encode_4d(coords)
        cur.execute("""
            INSERT INTO atoms (id, geometry, raw_value, modality)
            VALUES (%s, ST_MakePoint(%s, %s, %s, %s), %s, 'byte')
            ON CONFLICT (id) DO NOTHING
        """, (hilbert_id, coords[0], coords[1], coords[2], coords[3], bytes([byte_val])))
        byte_atoms[byte_val] = hilbert_id
    
    conn.commit()
    print(f"  ✓ {len(byte_atoms)} byte atoms")
    
    # Ingest tokens
    print("\n🔤 Ingesting token atoms...")
    for idx, token in enumerate(tokens[:10000]):
        if isinstance(token, str):
            coords = compute_4d_coordinates(token, 'token', idx, len(tokens))
            hilbert_id = hilbert_encode_4d(coords)
            
            cur.execute("""
                INSERT INTO atoms (id, geometry, raw_value, modality)
                VALUES (%s, ST_MakePoint(%s, %s, %s, %s), %s, 'token')
                ON CONFLICT (id) DO NOTHING
            """, (hilbert_id, coords[0], coords[1], coords[2], coords[3], token.encode('utf-8')))
            
            # Create relations
            for pos, byte_val in enumerate(token.encode('utf-8')):
                cur.execute("""
                    INSERT INTO relations (parent_id, child_id, relation_type, position)
                    VALUES (%s, %s, 'composition', %s)
                    ON CONFLICT DO NOTHING
                """, (hilbert_id, byte_atoms[byte_val], pos))
        
        if (idx + 1) % 1000 == 0:
            conn.commit()
            print(f"  Progress: {idx + 1}/10000")
    
    conn.commit()
    
    # Stats
    cur.execute("SELECT modality, COUNT(*) FROM atoms GROUP BY modality")
    print(f"\n📊 Final Statistics:")
    for modality, count in cur.fetchall():
        print(f"  {modality}: {count:,} atoms")
    
    cur.execute("SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type")
    print(f"\n🔗 Relations:")
    for rel_type, count in cur.fetchall():
        print(f"  {rel_type}: {count:,}")
    
    conn.close()
    print("\n✅ Complete!")

if __name__ == '__main__':
    main()
