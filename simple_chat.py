#!/usr/bin/env python3
"""
Simple working chat interface using Hartonomous geometric database.
Uses actual connections between atoms for next-token prediction.
"""

import psycopg2
import sys

DB_CONFIG = {
    'dbname': 'hartonomous',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost'
}

def generate_response(prompt, max_tokens=20):
    """Generate response using connection-based inference."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Tokenize prompt (simple word-based for now)
    tokens = prompt.split()
    response_tokens = []
    
    current_token = tokens[-1] if tokens else "Hello"
    
    for _ in range(max_tokens):
        # Find atom ID for current token
        cur.execute("SELECT id FROM atoms WHERE raw_value = %s", (current_token.encode(),))
        result = cur.fetchone()
        
        if not result:
            # Token not found, try spatial nearest neighbor
            break
        
        current_id = result[0]
        
        # Find next token via connections (highest weight first)
        cur.execute("""
            SELECT a.id, encode(a.raw_value, 'escape')::text as token, c.weight_id
            FROM connections c
            JOIN atoms a ON a.id = c.to_id
            WHERE c.from_id = %s
            ORDER BY c.weight_id DESC
            LIMIT 1
        """, (current_id,))
        
        next_token_result = cur.fetchone()
        
        if not next_token_result:
            # No more connections
            break
        
        next_id, next_token, weight = next_token_result
        response_tokens.append(next_token)
        current_token = next_token
        
        # Stop at sentence end
        if next_token in ['.', '!', '?']:
            break
    
    conn.close()
    return ' '.join(response_tokens)

def main():
    print("="*80)
    print("HARTONOMOUS CHAT - SQL-Based LLM Inference")
    print("="*80)
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Show database stats
    cur.execute("SELECT COUNT(*) FROM atoms")
    atom_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM connections")
    conn_count = cur.fetchone()[0]
    
    print(f"\nDatabase: {atom_count:,} atoms, {conn_count:,} connections")
    print(f"Inference: Pure PostgreSQL spatial queries (no GPU)\n")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if not prompt or prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            response = generate_response(prompt)
            
            if response:
                print(f"AI: {prompt} {response}\n")
            else:
                print("AI: [No connections found for this input]\n")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    conn.close()
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
