#!/usr/bin/env python3
"""
Hartonomous - Real Inference Using Geometric Atoms
Uses the actual ingested model data to generate text.
"""
import psycopg2
import numpy as np
import sys

class HartonomousLLM:
    def __init__(self):
        self.conn = psycopg2.connect(
            database='hartonomous',
            user='postgres',
            host='localhost',
            password='postgres'
        )
        self.cur = self.conn.cursor()
        
    def get_token_embedding(self, token_text):
        """Get embedding vector for a token from token_embeddings table."""
        self.cur.execute("""
            SELECT embedding_vector 
            FROM token_embeddings 
            WHERE token_text = %s
            LIMIT 1
        """, (token_text,))
        result = self.cur.fetchone()
        if result:
            return np.frombuffer(result[0], dtype=np.float16)
        return None
    
    def find_similar_tokens(self, embedding, k=5):
        """Find k nearest tokens by embedding similarity."""
        # Use PostgreSQL to compute cosine similarity
        self.cur.execute("""
            WITH target AS (
                SELECT %s::bytea as emb
            )
            SELECT 
                te.token_text,
                -- Cosine similarity would go here, but let's use spatial distance for now
                ST_3DDistance(
                    a.geom,
                    (SELECT geom FROM atoms WHERE id = te.atom_id LIMIT 1)
                ) as distance
            FROM token_embeddings te
            JOIN atoms a ON a.id = te.atom_id
            CROSS JOIN target
            WHERE te.token_text IS NOT NULL
            ORDER BY distance ASC
            LIMIT %s
        """, (embedding.tobytes(), k))
        return [row[0] for row in self.cur.fetchall()]
    
    def predict_next_via_compositions(self, current_token, k=10):
        """
        Find what tokens typically follow current_token by looking at
        composition sequences in the database.
        """
        self.cur.execute("""
            WITH current_atom AS (
                SELECT atom_id
                FROM token_embeddings
                WHERE token_text = %s
                LIMIT 1
            ),
            -- Find all compositions where current token appears
            seqs_with_current AS (
                SELECT DISTINCT parent_id, position
                FROM compositions c
                JOIN current_atom ca ON c.child_id = ca.atom_id
            ),
            -- Get the next token in each sequence
            next_tokens AS (
                SELECT 
                    c2.child_id as next_atom_id,
                    COUNT(*) as frequency
                FROM seqs_with_current swc
                JOIN compositions c2 ON c2.parent_id = swc.parent_id
                    AND c2.position = swc.position + 1
                GROUP BY c2.child_id
            )
            SELECT 
                te.token_text,
                nt.frequency
            FROM next_tokens nt
            JOIN token_embeddings te ON te.atom_id = nt.next_atom_id
            ORDER BY nt.frequency DESC
            LIMIT %s
        """, (current_token, k))
        
        results = self.cur.fetchall()
        return [r[0] for r in results] if results else []
    
    def predict_next_via_connections(self, current_token, k=5):
        """Use weight connections table to predict next token."""
        self.cur.execute("""
            WITH current_atom AS (
                SELECT atom_id
                FROM token_embeddings
                WHERE token_text = %s
                LIMIT 1
            )
            SELECT 
                te.token_text,
                w.value_float as weight
            FROM connections conn
            JOIN current_atom ca ON conn.from_id = ca.atom_id
            JOIN atoms w ON w.id = conn.weight_id
            JOIN token_embeddings te ON te.atom_id = conn.to_id
            ORDER BY w.value_float DESC NULLS LAST
            LIMIT %s
        """, (current_token, k))
        
        results = self.cur.fetchall()
        return [r[0] for r in results] if results else []
    
    def predict_next_token(self, current_token):
        """Predict next token using available methods."""
        # Try composition-based prediction first
        predictions = self.predict_next_via_compositions(current_token, k=5)
        
        if predictions:
            return predictions[0]
        
        # Fall back to connection-based
        predictions = self.predict_next_via_connections(current_token, k=5)
        
        if predictions:
            return predictions[0]
        
        # Fall back to geometric proximity
        embedding = self.get_token_embedding(current_token)
        if embedding is not None:
            similar = self.find_similar_tokens(embedding, k=5)
            if similar and len(similar) > 1:
                return similar[1]  # Return second similar (first is itself)
        
        return None
    
    def generate(self, prompt, max_tokens=30):
        """Generate text autoregressively."""
        tokens = prompt.strip().split()
        generated = []
        
        current = tokens[-1] if tokens else None
        
        for i in range(max_tokens):
            if not current:
                break
            
            next_tok = self.predict_next_token(current)
            
            if not next_tok or next_tok == current:
                break
            
            # Clean up BPE tokens
            display_token = next_tok.replace('Ġ', ' ')
            generated.append(display_token)
            current = next_tok
            
            # Stop on repetition
            if len(generated) > 3:
                if generated[-1] == generated[-3]:
                    break
        
        return ''.join(generated)
    
    def close(self):
        self.conn.close()

def main():
    print("=" * 70)
    print("  HARTONOMOUS - PostgreSQL Geometric LLM")
    print("  Using compositions and connections for inference")
    print("=" * 70)
    print()
    
    llm = HartonomousLLM()
    
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
        print(f"Prompt: {prompt}")
        response = llm.generate(prompt, max_tokens=50)
        print(f"Response: {prompt}{response}")
    else:
        print("Interactive mode. Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                response = llm.generate(user_input, max_tokens=50)
                print(f"Hartonomous: {user_input}{response}\n")
                
            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                print(f"Error: {e}\n")
                import traceback
                traceback.print_exc()
    
    llm.close()
    print("\nGoodbye!")

if __name__ == '__main__':
    main()
