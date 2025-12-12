"""Complete model ingestion from SafeTensors"""
import struct
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import json

try:
    from safetensors import safe_open
    import torch
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'safetensors', 'torch', '--quiet'])
    from safetensors import safe_open
    import torch

from ..db import Database


class HilbertEncoder:
    """4D Hilbert curve encoding for spatial indexing"""
    
    @staticmethod
    def encode(x: float, y: float, z: float, m: float, bits: int = 16) -> int:
        """Encode 4D coordinates [-1,1]^4 to Hilbert index"""
        max_val = (1 << bits) - 1
        
        # Normalize to [0, max_val]
        ix = int((x + 1) * 0.5 * max_val) & max_val
        iy = int((y + 1) * 0.5 * max_val) & max_val
        iz = int((z + 1) * 0.5 * max_val) & max_val
        im = int((m + 1) * 0.5 * max_val) & max_val
        
        # Bit-interleave
        result = 0
        for i in range(bits):
            result |= ((ix >> i) & 1) << (i * 4 + 0)
            result |= ((iy >> i) & 1) << (i * 4 + 1)
            result |= ((iz >> i) & 1) << (i * 4 + 2)
            result |= ((im >> i) & 1) << (i * 4 + 3)
        
        return result


class ModelIngester:
    """Ingest complete model: vocab, embeddings, weights"""
    
    def __init__(self, db: Database, model_dir: Path):
        self.db = db
        self.model_dir = Path(model_dir)
        
        # Caches for deduplication
        self.float_cache: Dict[bytes, int] = {}  # raw_bytes -> atom_id
        
        # Stats
        self.stats = {
            'atoms_created': 0,
            'atoms_reused': 0,
            'relations_created': 0,
            'tokens_processed': 0,
            'embeddings_processed': 0,
            'weights_processed': 0
        }
    
    def create_atom(self, raw_value: bytes, x: float, y: float, z: float, m: float) -> int:
        """Create or get existing atom"""
        # Check cache
        if raw_value in self.float_cache:
            self.stats['atoms_reused'] += 1
            return self.float_cache[raw_value]
        
        # Generate Hilbert ID
        hilbert_id = HilbertEncoder.encode(x, y, z, m)
        
        # Insert atom
        self.db.execute("""
            INSERT INTO atoms (id, geom, raw_value)
            VALUES (%s, ST_MakePointZM(%s, %s, %s, %s), %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
        """, (hilbert_id, x, y, z, m, raw_value))
        
        result = self.db.fetchone()
        if result:
            atom_id = result[0]
            self.stats['atoms_created'] += 1
        else:
            # Already exists, fetch it
            self.db.execute("SELECT id FROM atoms WHERE id = %s", (hilbert_id,))
            atom_id = self.db.fetchone()[0]
            self.stats['atoms_reused'] += 1
        
        self.float_cache[raw_value] = atom_id
        return atom_id
    
    def ingest_vocabulary(self):
        """Ingest token vocabulary"""
        print("\n📚 Ingesting vocabulary...")
        
        vocab_path = self.model_dir / 'vocab.json'
        vocab = json.load(open(vocab_path))
        
        vocab_size = len(vocab)
        print(f"  Vocab size: {vocab_size:,}")
        
        batch = []
        for token_text, token_id in tqdm(vocab.items(), desc="Tokens"):
            # Store in token_embeddings table
            batch.append((token_id, token_text, None, None))
            
            if len(batch) >= 1000:
                self.db.execute("BEGIN")
                self.db.batch_insert("""
                    INSERT INTO token_embeddings (token_id, token_text, embedding_atom_id, embedding_h_coords)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (token_id) DO NOTHING
                """, batch, page_size=1000)
                self.db.commit()
                batch = []
                self.stats['tokens_processed'] += 1000
        
        if batch:
            self.db.execute("BEGIN")
            self.db.batch_insert("""
                INSERT INTO token_embeddings (token_id, token_text, embedding_atom_id, embedding_h_coords)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (token_id) DO NOTHING
            """, batch)
            self.db.commit()
            self.stats['tokens_processed'] += len(batch)
        
        print(f"  ✓ {self.stats['tokens_processed']:,} tokens")
    
    def ingest_embeddings(self):
        """Ingest token embedding matrix"""
        print("\n🔢 Ingesting embeddings...")
        
        safetensors_path = self.model_dir / 'model.safetensors'
        
        with safe_open(str(safetensors_path), framework='pt') as f:
            # Find embedding tensor
            emb_key = None
            for key in f.keys():
                if 'embed_tokens' in key:
                    emb_key = key
                    break
            
            if not emb_key:
                print("  ❌ No embedding tensor found")
                return
            
            print(f"  Loading: {emb_key}")
            emb_tensor_raw = f.get_tensor(emb_key)
            # Convert bfloat16 to float32, then to numpy
            emb_tensor = emb_tensor_raw.to(torch.float32).cpu().numpy()
            vocab_size, emb_dim = emb_tensor.shape
            print(f"  Shape: ({vocab_size:,}, {emb_dim})")
            
            # Process each token embedding
            for token_id in tqdm(range(vocab_size), desc="Embeddings"):
                emb_vector = emb_tensor[token_id]
                
                # Create atoms for each dimension value
                atom_ids = []
                h_coords = []
                
                for dim_idx, value in enumerate(emb_vector):
                    # Create float atom
                    raw_bytes = struct.pack('<f', float(value))
                    
                    # 4D coords: (value, dim_position, token_position, reserved)
                    norm_val = np.tanh(float(value))  # Normalize value
                    norm_dim = (dim_idx / emb_dim) * 2 - 1  # [-1, 1]
                    norm_tok = (token_id / vocab_size) * 2 - 1  # [-1, 1]
                    
                    atom_id = self.create_atom(raw_bytes, norm_val, norm_dim, norm_tok, 0.0)
                    atom_ids.append(atom_id)
                    h_coords.append(atom_id)  # Use atom_id as Hilbert coord
                
                # Update token_embeddings with atom references
                self.db.execute("""
                    UPDATE token_embeddings
                    SET embedding_atom_id = %s,
                        embedding_h_coords = %s
                    WHERE token_id = %s
                """, (atom_ids, h_coords, token_id))
                
                self.stats['embeddings_processed'] += 1
                
                if token_id % 1000 == 0:
                    self.db.commit()
            
            self.db.commit()
            print(f"  ✓ {self.stats['embeddings_processed']:,} embeddings")
    
    def ingest_attention_layer(self, layer_num: int = 0):
        """Ingest attention weights for one layer"""
        print(f"\n⚡ Ingesting layer {layer_num} attention...")
        
        safetensors_path = self.model_dir / 'model.safetensors'
        
        with safe_open(str(safetensors_path), framework='pt') as f:
            # Find Q, K, V projection matrices
            for proj_type in ['q_proj', 'k_proj', 'v_proj']:
                key = f'model.layers.{layer_num}.self_attn.{proj_type}.weight'
                
                if key not in f.keys():
                    print(f"  ⚠️  {key} not found")
                    continue
                
                print(f"  Processing: {key}")
                tensor_raw = f.get_tensor(key)
                # Convert bfloat16 to float32, then to numpy
                tensor = tensor_raw.to(torch.float32).cpu().numpy()
                print(f"    Shape: {tensor.shape}")
                
                # Store sparse representation (top K values)
                flat = tensor.flatten()
                top_k = min(10000, len(flat))  # Store top 10K weights
                top_indices = np.argpartition(np.abs(flat), -top_k)[-top_k:]
                
                batch = []
                for flat_idx in tqdm(top_indices, desc=f"  {proj_type}", leave=False):
                    row, col = np.unravel_index(flat_idx, tensor.shape)
                    value = float(tensor[row, col])
                    
                    # Create weight atom
                    raw_bytes = struct.pack('<f', value)
                    weight_atom_id = self.create_atom(
                        raw_bytes,
                        np.tanh(value),  # x
                        (row / tensor.shape[0]) * 2 - 1,  # y
                        (col / tensor.shape[1]) * 2 - 1,  # z
                        0.0  # m
                    )
                    
                    # Create relation
                    batch.append((
                        row,  # parent_id (from neuron)
                        col,  # child_id (to neuron)
                        f'attention_{proj_type}',  # relation_type
                        None,  # position
                        weight_atom_id,  # weight_atom_id
                        f'layer_{layer_num}',  # layer_name
                        None  # meta
                    ))
                    
                    if len(batch) >= 1000:
                        self.db.batch_insert("""
                            INSERT INTO relations (parent_id, child_id, relation_type, position, weight_atom_id, layer_name, meta)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, batch)
                        self.db.commit()
                        self.stats['weights_processed'] += len(batch)
                        self.stats['relations_created'] += len(batch)
                        batch = []
                
                if batch:
                    self.db.batch_insert("""
                        INSERT INTO relations (parent_id, child_id, relation_type, position, weight_atom_id, layer_name, meta)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, batch)
                    self.db.commit()
                    self.stats['weights_processed'] += len(batch)
                    self.stats['relations_created'] += len(batch)
        
        print(f"  ✓ Layer {layer_num} attention weights")
    
    def print_stats(self):
        """Print ingestion statistics"""
        print("\n" + "="*60)
        print("INGESTION COMPLETE")
        print("="*60)
        for key, value in self.stats.items():
            print(f"{key:30s}: {value:>15,}")
        print("="*60)
    
    def run_full_ingestion(self):
        """Run complete model ingestion"""
        print("="*60)
        print("HARTONOMOUS MODEL INGESTION")
        print(f"Model: {self.model_dir}")
        print("="*60)
        
        self.ingest_vocabulary()
        self.ingest_embeddings()
        self.ingest_attention_layer(layer_num=0)
        
        self.print_stats()
