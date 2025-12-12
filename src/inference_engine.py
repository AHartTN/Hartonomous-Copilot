"""
Hartonomous Inference Engine
Pure PostgreSQL + PL/Python3u transformer inference
On-demand sparse queries - only fetch what we need, when we need it
"""

import psycopg2
import numpy as np
from typing import List, Tuple
import json

class HartonomousInference:
    def __init__(self, conn_string="dbname=hartonomous user=ahart"):
        self.conn = psycopg2.connect(conn_string)
        self.cur = self.conn.cursor()
        
    def get_token_embedding(self, token_id: int) -> np.ndarray:
        """Fetch embedding vector for a specific token on-demand"""
        self.cur.execute("""
            SELECT 
                r.position,
                a.raw_value
            FROM relations r
            JOIN atoms a ON a.id = r.child_id
            WHERE r.parent_id = %s 
              AND r.relation_type = 'embedding'
            ORDER BY r.position
        """, (token_id,))
        
        rows = self.cur.fetchall()
        if not rows:
            return np.zeros(896, dtype=np.float32)
            
        embedding = np.zeros(len(rows), dtype=np.float32)
        for pos, raw_bytes in rows:
            embedding[pos] = np.frombuffer(raw_bytes, dtype=np.float32)[0]
        return embedding
    
    def get_attention_weights(self, layer: int, head: int, query_pos: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fetch Q, K, V weight matrices for specific layer/head only when needed"""
        # Sparse query - only get weights for this specific attention head
        self.cur.execute("""
            SELECT 
                r.meta->>'matrix_type' as matrix_type,
                r.position,
                a.raw_value
            FROM relations r
            JOIN atoms a ON a.id = r.child_id
            WHERE r.relation_type = 'attention_weight'
              AND r.meta->>'layer' = %s
              AND r.meta->>'head' = %s
            ORDER BY matrix_type, position
        """, (str(layer), str(head)))
        
        rows = self.cur.fetchall()
        
        q_weights = []
        k_weights = []
        v_weights = []
        
        for matrix_type, pos, raw_bytes in rows:
            val = np.frombuffer(raw_bytes, dtype=np.float32)[0]
            if matrix_type == 'Q':
                q_weights.append((pos, val))
            elif matrix_type == 'K':
                k_weights.append((pos, val))
            elif matrix_type == 'V':
                v_weights.append((pos, val))
        
        # Convert to dense arrays (or keep sparse if implementing sparse ops)
        dim = 896
        Q = np.zeros((dim, dim), dtype=np.float32)
        K = np.zeros((dim, dim), dtype=np.float32)
        V = np.zeros((dim, dim), dtype=np.float32)
        
        for pos, val in q_weights:
            row, col = divmod(pos, dim)
            Q[row, col] = val
        for pos, val in k_weights:
            row, col = divmod(pos, dim)
            K[row, col] = val
        for pos, val in v_weights:
            row, col = divmod(pos, dim)
            V[row, col] = val
            
        return Q, K, V
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def attention_forward(self, embeddings: np.ndarray, layer: int, head: int) -> np.ndarray:
        """
        Single attention head forward pass
        embeddings: [seq_len, hidden_dim]
        """
        Q, K, V = self.get_attention_weights(layer, head)
        
        # Q = embeddings @ Q.T  [seq_len, dim]
        # K = embeddings @ K.T  [seq_len, dim]
        # V = embeddings @ V.T  [seq_len, dim]
        q = embeddings @ Q.T
        k = embeddings @ K.T
        v = embeddings @ V.T
        
        # Scaled dot-product attention
        scores = (q @ k.T) / np.sqrt(q.shape[-1])
        attn_weights = self.softmax(scores)
        output = attn_weights @ v
        
        return output
    
    def layer_forward(self, hidden_states: np.ndarray, layer: int) -> np.ndarray:
        """
        Full transformer layer forward pass
        Multi-head attention + FFN
        """
        # Multi-head attention (simplified - should concat all heads)
        attn_output = self.attention_forward(hidden_states, layer, head=0)
        
        # Residual connection + LayerNorm (simplified)
        hidden_states = hidden_states + attn_output
        
        # Feed-forward network (query weights on-demand)
        self.cur.execute("""
            SELECT 
                r.position,
                a.raw_value
            FROM relations r
            JOIN atoms a ON a.id = r.child_id
            WHERE r.relation_type = 'ffn_weight'
              AND r.meta->>'layer' = %s
            ORDER BY position
            LIMIT 1000  -- Get sample for now
        """, (str(layer),))
        
        # FFN implementation would go here
        # For now, just return the attention output
        
        return hidden_states
    
    def predict_next_token(self, input_ids: List[int], max_new_tokens: int = 20) -> List[int]:
        """
        Autoregressive generation
        Only queries DB for what we need, when we need it
        """
        output_ids = input_ids.copy()
        
        for _ in range(max_new_tokens):
            # Get embeddings for current sequence
            seq_len = len(output_ids)
            embeddings = np.zeros((seq_len, 896), dtype=np.float32)
            
            for i, token_id in enumerate(output_ids):
                embeddings[i] = self.get_token_embedding(token_id)
            
            # Forward pass through layers (on-demand weight loading)
            hidden_states = embeddings
            
            # Process through transformer layers
            num_layers = 24  # Qwen2.5 has 24 layers
            for layer in range(num_layers):
                hidden_states = self.layer_forward(hidden_states, layer)
                print(f"Layer {layer} complete", end='\r')
            
            # Get logits for last position
            last_hidden = hidden_states[-1]
            
            # Project to vocabulary (query final layer weights)
            self.cur.execute("""
                SELECT 
                    a.id as token_id,
                    ST_Distance(
                        a.geometry,
                        ST_MakePoint(%s, %s, %s, %s)::geometry
                    ) as distance
                FROM atoms a
                WHERE a.modality = 'token'
                ORDER BY distance
                LIMIT 10
            """, (float(last_hidden[0]), float(last_hidden[1]), 
                  float(last_hidden[2]), float(last_hidden[3])))
            
            candidates = self.cur.fetchall()
            if not candidates:
                break
            
            # Pick most likely token (closest in embedding space)
            next_token_id = candidates[0][0]
            output_ids.append(next_token_id)
            
            # Decode and print
            self.cur.execute("SELECT raw_value FROM atoms WHERE id = %s", (next_token_id,))
            token_bytes = self.cur.fetchone()[0]
            token_str = token_bytes.decode('utf-8', errors='ignore')
            print(f"\nGenerated: {token_str}")
            
            # Stop on end token
            if token_str in ['</s>', '<|endoftext|>']:
                break
        
        return output_ids
    
    def chat(self, prompt: str) -> str:
        """Interactive chat interface"""
        # Tokenize prompt (lookup tokens in DB)
        tokens = []
        for char in prompt:
            self.cur.execute("""
                SELECT id FROM atoms 
                WHERE modality = 'token' 
                  AND raw_value = %s
                LIMIT 1
            """, (char.encode('utf-8'),))
            result = self.cur.fetchone()
            if result:
                tokens.append(result[0])
        
        if not tokens:
            return "Could not tokenize input"
        
        print(f"Input tokens: {tokens}")
        output_ids = self.predict_next_token(tokens, max_new_tokens=50)
        
        # Decode output
        response_parts = []
        for token_id in output_ids[len(tokens):]:
            self.cur.execute("SELECT raw_value FROM atoms WHERE id = %s", (token_id,))
            result = self.cur.fetchone()
            if result:
                response_parts.append(result[0].decode('utf-8', errors='ignore'))
        
        return ''.join(response_parts)

if __name__ == '__main__':
    engine = HartonomousInference()
    
    print("=" * 60)
    print("  HARTONOMOUS INFERENCE ENGINE")
    print("  On-Demand Sparse Queries | No GPU Required")
    print("=" * 60)
    
    # Test embedding retrieval
    print("\n[TEST 1] Fetching token embedding...")
    embedding = engine.get_token_embedding(1)
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 10 values: {embedding[:10]}")
    
    # Test chat
    print("\n[TEST 2] Chat interface...")
    response = engine.chat("Hello")
    print(f"Response: {response}")
    
    engine.conn.close()
