#!/usr/bin/env python3
"""
Unified Graph Ingestion: Ingest entire GGUF model as atomic graph
"""
import sys
import struct
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

@dataclass
class GGUFTensor:
    name: str
    shape: List[int]
    dtype: str
    offset: int
    data: np.ndarray = None

class UnifiedGraphIngester:
    """Ingest GGUF model as pure atomic graph"""
    
    def __init__(self, db_conn, model_path: Path):
        self.conn = db_conn
        self.cur = db_conn.cursor()
        self.model_path = model_path
        
        # Atom caches for deduplication
        self.float_atoms: Dict[float, int] = {}  # value -> atom_id
        self.token_atoms: Dict[bytes, int] = {}  # token_bytes -> atom_id
        self.layer_atoms: Dict[str, int] = {}    # layer_name -> atom_id
        
        # Stats
        self.stats = {
            'tokens_created': 0,
            'floats_created': 0,
            'floats_reused': 0,
            'embeddings_created': 0,
            'weights_created': 0
        }
    
    def create_atom(self, raw_value: bytes, geometry_coords: Tuple[float, float, float, float]) -> int:
        """Create or retrieve atom by content hash"""
        content_hash = hashlib.sha256(raw_value).digest()
        
        # Check if exists
        self.cur.execute(
            "SELECT id FROM atoms WHERE content_hash = %s",
            (content_hash,)
        )
        result = self.cur.fetchone()
        if result:
            return result[0]
        
        # Create Hilbert coordinate from 4D position
        x, y, z, w = geometry_coords
        hilbert_val = self.hilbert_encode_4d(x, y, z, w)
        
        # Insert atom
        self.cur.execute("""
            INSERT INTO atoms (id, geometry, raw_value, content_hash)
            VALUES (%s, ST_MakePoint(%s, %s, %s), %s, %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
        """, (hilbert_val, x, y, z, raw_value, content_hash))
        
        result = self.cur.fetchone()
        return result[0] if result else hilbert_val
    
    def hilbert_encode_4d(self, x: float, y: float, z: float, w: float, bits: int = 16) -> int:
        """Encode 4D coordinates to Hilbert curve value"""
        # Normalize to [0, 2^bits)
        max_val = (1 << bits) - 1
        ix = int((x + 1) * 0.5 * max_val)
        iy = int((y + 1) * 0.5 * max_val)
        iz = int((z + 1) * 0.5 * max_val)
        iw = int((w + 1) * 0.5 * max_val)
        
        # Simple 4D Hilbert encoding (interleaved bits)
        result = 0
        for i in range(bits):
            result |= ((ix >> i) & 1) << (i * 4 + 0)
            result |= ((iy >> i) & 1) << (i * 4 + 1)
            result |= ((iz >> i) & 1) << (i * 4 + 2)
            result |= ((iw >> i) & 1) << (i * 4 + 3)
        
        return result
    
    def get_or_create_float_atom(self, value: float) -> int:
        """Get or create atom for float value"""
        # Round to handle floating point precision
        rounded = round(value, 8)
        
        if rounded in self.float_atoms:
            self.stats['floats_reused'] += 1
            return self.float_atoms[rounded]
        
        # Create new float atom
        raw_bytes = struct.pack('f', value)
        
        # Project float to 4D space: (value, |value|, sign, 0)
        coords = (
            np.tanh(value),  # x: normalized value
            abs(value) / (abs(value) + 1),  # y: magnitude
            1.0 if value >= 0 else -1.0,  # z: sign
            0.0  # w: reserved
        )
        
        atom_id = self.create_atom(raw_bytes, coords)
        self.float_atoms[rounded] = atom_id
        self.stats['floats_created'] += 1
        
        return atom_id
    
    def get_or_create_token_atom(self, token_bytes: bytes, token_id: int, vocab_size: int) -> int:
        """Get or create atom for token"""
        if token_bytes in self.token_atoms:
            return self.token_atoms[token_bytes]
        
        # Project token to 4D space based on token_id position in vocabulary
        t = token_id / vocab_size  # Normalize to [0, 1]
        coords = (
            np.cos(2 * np.pi * t),  # x: circular embedding
            np.sin(2 * np.pi * t),  # y: circular embedding
            t * 2 - 1,  # z: linear position
            0.0  # w: reserved for modality
        )
        
        atom_id = self.create_atom(token_bytes, coords)
        self.token_atoms[token_bytes] = atom_id
        self.stats['tokens_created'] += 1
        
        return atom_id
    
    def read_gguf_header(self, f) -> Tuple[Dict, List[GGUFTensor]]:
        """Read GGUF file header and tensor metadata"""
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Not a GGUF file: {magic}")
        
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
        
        # Read metadata
        metadata = {}
        for _ in range(metadata_kv_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            
            value_type = struct.unpack('<I', f.read(4))[0]
            
            # Parse value based on type
            if value_type == 4:  # String
                val_len = struct.unpack('<Q', f.read(8))[0]
                value = f.read(val_len).decode('utf-8')
            elif value_type == 8:  # Array
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                value = []
                for _ in range(arr_len):
                    if arr_type == 4:  # String array
                        str_len = struct.unpack('<Q', f.read(8))[0]
                        value.append(f.read(str_len).decode('utf-8'))
            else:
                # Skip other types for now
                value = None
            
            metadata[key] = value
        
        # Read tensor info
        tensors = []
        for _ in range(tensor_count):
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            
            n_dims = struct.unpack('<I', f.read(4))[0]
            shape = list(struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims)))
            
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            
            tensors.append(GGUFTensor(name, shape, dtype, offset))
        
        return metadata, tensors
    
    def ingest_vocabulary(self, metadata: Dict):
        """Ingest vocabulary tokens as atoms"""
        print("\n📚 Ingesting Vocabulary...")
        
        tokens = metadata.get('tokenizer.ggml.tokens', [])
        print(f"Found {len(tokens)} tokens in vocabulary")
        
        for token_id, token in enumerate(tqdm(tokens, desc="Creating token atoms")):
            token_bytes = token.encode('utf-8')
            self.get_or_create_token_atom(token_bytes, token_id, len(tokens))
        
        self.conn.commit()
        print(f"✅ Created {self.stats['tokens_created']} token atoms")
    
    def ingest_embeddings(self, model_file, tensor: GGUFTensor):
        """Ingest embedding layer as token->float relations"""
        print(f"\n🔗 Ingesting Embeddings: {tensor.name}")
        print(f"Shape: {tensor.shape}")
        
        # Read tensor data
        model_file.seek(tensor.offset)
        
        vocab_size, embed_dim = tensor.shape
        
        # Read all embeddings
        dtype_map = {0: np.float32, 1: np.float16}
        dt = dtype_map.get(tensor.dtype, np.float32)
        
        embeddings = np.fromfile(model_file, dtype=dt, count=vocab_size * embed_dim)
        embeddings = embeddings.reshape(vocab_size, embed_dim)
        
        print(f"Loaded {embeddings.shape} embedding matrix")
        
        # Create relations batch
        relations_batch = []
        
        for token_id in tqdm(range(vocab_size), desc="Creating embedding relations"):
            token_bytes = list(self.token_atoms.keys())[token_id] if token_id < len(self.token_atoms) else b''
            if not token_bytes:
                continue
            
            token_atom_id = self.token_atoms[token_bytes]
            
            for dim_idx in range(embed_dim):
                value = float(embeddings[token_id, dim_idx])
                
                # Get or create float atom
                float_atom_id = self.get_or_create_float_atom(value)
                
                # Create relation: token -> embedding_value
                relations_batch.append((
                    token_atom_id,  # parent
                    float_atom_id,  # child
                    'embedding',  # type
                    None,  # weight_atom_id (value is the child itself)
                    dim_idx,  # position
                    tensor.name,  # layer_name
                    None  # meta
                ))
                
                self.stats['embeddings_created'] += 1
            
            # Batch insert every 10k relations
            if len(relations_batch) >= 10000:
                execute_batch(self.cur, """
                    INSERT INTO relations (parent_id, child_id, relation_type, weight_atom_id, position, layer_name, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, relations_batch)
                self.conn.commit()
                relations_batch = []
        
        # Insert remaining
        if relations_batch:
            execute_batch(self.cur, """
                INSERT INTO relations (parent_id, child_id, relation_type, weight_atom_id, position, layer_name, meta)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, relations_batch)
            self.conn.commit()
        
        print(f"✅ Created {self.stats['embeddings_created']} embedding relations")
    
    def ingest_attention_weights(self, model_file, tensors: List[GGUFTensor], max_weights: int = 100000):
        """Ingest attention weight matrices as sparse graph"""
        print(f"\n🧠 Ingesting Attention Weights (sparse, top {max_weights} per layer)...")
        
        attention_tensors = [t for t in tensors if 'attn' in t.name and 'weight' in t.name]
        
        for tensor in attention_tensors[:5]:  # First 5 layers to start
            print(f"\nLayer: {tensor.name}, Shape: {tensor.shape}")
            
            model_file.seek(tensor.offset)
            
            # Read weights
            total_size = np.prod(tensor.shape)
            dtype_map = {0: np.float32, 1: np.float16}
            dt = dtype_map.get(tensor.dtype, np.float32)
            
            weights = np.fromfile(model_file, dtype=dt, count=total_size)
            weights = weights.reshape(tensor.shape)
            
            # Get top-k connections by magnitude
            flat_weights = weights.flatten()
            top_indices = np.argsort(np.abs(flat_weights))[-max_weights:]
            
            relations_batch = []
            
            for flat_idx in tqdm(top_indices, desc=f"Creating {tensor.name} relations"):
                # Convert flat index to 2D
                if len(tensor.shape) == 2:
                    row, col = np.unravel_index(flat_idx, tensor.shape)
                    value = float(weights[row, col])
                    
                    # Create atoms for indices (representing neurons)
                    from_neuron = self.get_or_create_float_atom(float(row))
                    to_neuron = self.get_or_create_float_atom(float(col))
                    weight_atom = self.get_or_create_float_atom(value)
                    
                    relations_batch.append((
                        from_neuron,
                        to_neuron,
                        'attention_weight',
                        weight_atom,
                        None,
                        tensor.name,
                        json.dumps({'row': int(row), 'col': int(col)})
                    ))
                    
                    self.stats['weights_created'] += 1
                
                # Batch insert
                if len(relations_batch) >= 1000:
                    execute_batch(self.cur, """
                        INSERT INTO relations (parent_id, child_id, relation_type, weight_atom_id, position, layer_name, meta)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, relations_batch)
                    self.conn.commit()
                    relations_batch = []
            
            if relations_batch:
                execute_batch(self.cur, """
                    INSERT INTO relations (parent_id, child_id, relation_type, weight_atom_id, position, layer_name, meta)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, relations_batch)
                self.conn.commit()
        
        print(f"✅ Created {self.stats['weights_created']} weight relations")
    
    def run(self):
        """Execute full unified ingestion"""
        print(f"🚀 Starting Unified Graph Ingestion: {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            # Read GGUF structure
            metadata, tensors = self.read_gguf_header(f)
            
            print(f"\n📊 Model Info:")
            print(f"  Tensors: {len(tensors)}")
            print(f"  Vocab Size: {len(metadata.get('tokenizer.ggml.tokens', []))}")
            
            # Phase 1: Vocabulary
            self.ingest_vocabulary(metadata)
            
            # Phase 2: Embeddings
            embedding_tensor = next((t for t in tensors if 'token_embd' in t.name), None)
            if embedding_tensor:
                self.ingest_embeddings(f, embedding_tensor)
            
            # Phase 3: Attention Weights (sparse)
            self.ingest_attention_weights(f, tensors)
        
        # Final stats
        print(f"\n" + "="*60)
        print("📈 Ingestion Complete!")
        print(f"  Token Atoms: {self.stats['tokens_created']}")
        print(f"  Float Atoms Created: {self.stats['floats_created']}")
        print(f"  Float Atoms Reused: {self.stats['floats_reused']}")
        print(f"  Deduplication Ratio: {self.stats['floats_reused'] / max(self.stats['floats_created'], 1):.2f}x")
        print(f"  Embedding Relations: {self.stats['embeddings_created']}")
        print(f"  Weight Relations: {self.stats['weights_created']}")
        print("="*60)

def main():
    model_path = Path("/usr/share/ollama/.ollama/models/blobs/sha256-c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    # Connect to database
    conn = psycopg2.connect(
        dbname="hartonomous",
        user="postgres",
        host="localhost",
        password=""
    )
    
    try:
        ingester = UnifiedGraphIngester(conn, model_path)
        ingester.run()
    finally:
        conn.close()

if __name__ == '__main__':
    main()
