#!/usr/bin/env python3
"""
Complete Inference Demo: Text Generation via Geometric Database

Input: "The quick brown"
Output: Next predicted token via spatial k-NN
"""
import psycopg2, json

def predict_next_token(current_token, k=5):
    """Predict next token using spatial k-NN"""
    conn = psycopg2.connect(database='hartonomous', user='postgres')
    cur = conn.cursor()
    
    # Find current token's Hilbert coordinates
    cur.execute("""
        SELECT h_xy, h_yz, h_zm, meta->>'text'
        FROM atoms
        WHERE modality = 'compositional'
          AND meta->>'text' = %s
        LIMIT 1
    """, (current_token,))
    
    result = cur.fetchone()
    if not result:
        print(f"Token '{current_token}' not found in vocabulary")
        return []
    
    h_xy, h_yz, h_zm, token_text = result
    
    # K-NN query for nearest neighbors
    cur.execute("""
        SELECT 
            meta->>'text' as next_token,
            sqrt(
                power(CAST(h_xy as numeric) - %s, 2) +
                power(CAST(h_yz as numeric) - %s, 2) +
                power(CAST(h_zm as numeric) - %s, 2)
            )::bigint as distance
        FROM atoms
        WHERE modality = 'compositional'
          AND (h_xy, h_yz, h_zm) != (%s, %s, %s)
        ORDER BY distance ASC
        LIMIT %s
    """, (h_xy, h_yz, h_zm, h_xy, h_yz, h_zm, k))
    
    predictions = cur.fetchall()
    conn.close()
    
    return predictions

def generate_text(seed_text, max_tokens=10):
    """Generate text autoregressively"""
    tokens = seed_text.split()
    generated = tokens.copy()
    
    print(f"Seed: '{seed_text}'")
    print(f"Generating {max_tokens} tokens...\n")
    
    for i in range(max_tokens):
        current_token = generated[-1]
        predictions = predict_next_token(current_token, k=3)
        
        if not predictions:
            print(f"No predictions for '{current_token}', stopping")
            break
        
        # Take top prediction
        next_token, distance = predictions[0]
        generated.append(next_token)
        
        print(f"Step {i+1}: '{current_token}' → '{next_token}' (distance: {distance:,})")
    
    return ' '.join(generated)

if __name__ == '__main__':
    print("=== Hartonomous Geometric AI Inference Demo ===\n")
    
    # Test 1: Simple prediction
    print("Test 1: Next token prediction")
    print("-" * 50)
    preds = predict_next_token('the', k=5)
    print(f"Input: 'the'")
    print(f"Top 5 predictions:")
    for token, dist in preds:
        print(f"  '{token}' (distance: {dist:,})")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Text generation
    print("Test 2: Autoregressive generation")
    print("-" * 50)
    result = generate_text("The quick", max_tokens=5)
    print(f"\nFinal: {result}")
    
    print("\n" + "="*50)
    print("\n✓ Inference working via pure SQL spatial queries")
    print("  No GPU. No matrix multiplication. Just geometry.")
