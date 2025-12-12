#!/usr/bin/env python3
"""
Hartonomous Inference Engine
Uses geometric relations to perform transformer inference in PostgreSQL
"""
import psycopg2
import numpy as np
import json

def predict_next_token(input_text, max_tokens=20):
    """Generate text using geometric database inference."""
    conn = psycopg2.connect(dbname='hartonomous', user='ahart', password='ahart', host='/var/run/postgresql')
    cur = conn.cursor()
    
    # Tokenize input (simple whitespace for demo)
    tokens = input_text.strip().split()
    output_tokens = list(tokens)
    
    for _ in range(max_tokens):
        # Get last token
        last_token = output_tokens[-1]
        
        # Find token atom
        cur.execute("""
            SELECT id, raw_value 
            FROM atoms 
            WHERE raw_value = %s
            LIMIT 1
        """, (last_token.encode(),))
        
        result = cur.fetchone()
        if not result:
            break
        
        token_id = result[0]
        
        # Get embedding for this token
        cur.execute("""
            SELECT a.raw_value, r.position
            FROM relations r
            JOIN atoms a ON a.id = r.child_id
            WHERE r.parent_id = %s 
              AND r.relation_type = 'embedding'
            ORDER BY r.position
            LIMIT 50
        """, (token_id,))
        
        embedding_parts = cur.fetchall()
        if not embedding_parts:
            break
        
        # Build embedding vector
        embedding = np.array([np.frombuffer(ep[0], dtype=np.float32)[0] for ep in embedding_parts])
        
        # Find geometrically similar tokens
        cur.execute("""
            WITH current_token AS (
                SELECT geometry FROM atoms WHERE id = %s
            )
            SELECT a.id, a.raw_value, 
                   ST_3DDistance(a.geometry, ct.geometry) as dist
            FROM atoms a, current_token ct
            WHERE a.id != %s
              AND ST_GeometryType(a.geometry) = 'ST_Point'
            ORDER BY a.geometry <-> ct.geometry
            LIMIT 10
        """, (token_id, token_id))
        
        candidates = cur.fetchall()
        if not candidates:
            break
        
        # Pick closest token that's not a weight
        for cand_id, cand_raw, dist in candidates:
            try:
                token_str = cand_raw.decode('utf-8')
                if len(token_str) > 0 and ' ' not in token_str:
                    output_tokens.append(token_str)
                    break
            except:
                continue
        else:
            break
    
    cur.close()
    conn.close()
    
    return ' '.join(output_tokens)

if __name__ == '__main__':
    print("=== HARTONOMOUS INFERENCE ===\n")
    
    prompts = [
        "Hello",
        "The quick",
        "I am"
    ]
    
    for prompt in prompts:
        result = predict_next_token(prompt, max_tokens=5)
        print(f"Input:  {prompt}")
        print(f"Output: {result}\n")
