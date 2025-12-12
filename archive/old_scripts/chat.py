#!/usr/bin/env python3
"""
Hartonomous Chat Interface - LLM powered by PostgreSQL geometry
"""
import psycopg2
import sys

class HartonomousLLM:
    def __init__(self):
        self.conn = psycopg2.connect(
            database='hartonomous',
            user='postgres',
            password='postgres',
            host='localhost'
        )
        self.cur = self.conn.cursor()
        
    def predict_next_token(self, current_token, k=1):
        """Predict next token via k-NN spatial search"""
        self.cur.execute("""
            WITH target AS (
                SELECT id, geom
                FROM atoms 
                WHERE raw_value = %s::bytea
                LIMIT 1
            )
            SELECT 
                a.raw_value::text as token,
                ST_3DDistance(a.geom, t.geom) as distance
            FROM atoms a, target t
            WHERE a.id != t.id
              AND a.raw_value IS NOT NULL
            ORDER BY a.geom <-> t.geom
            LIMIT %s
        """, (current_token.encode(), k))
        
        results = self.cur.fetchall()
        return [r[0] for r in results] if results else []
    
    def generate(self, prompt, max_tokens=50, temperature=0.0):
        """Generate text autoregressively"""
        tokens = prompt.split()
        generated = tokens.copy()
        
        for i in range(max_tokens):
            current = generated[-1]
            
            # Get top-k predictions
            k = 1 if temperature == 0.0 else 5
            predictions = self.predict_next_token(current, k=k)
            
            if not predictions:
                break
            
            # Take first (greedy) for now
            next_token = predictions[0]
            generated.append(next_token)
            
            # Stop on repetition
            if len(generated) > 2 and next_token == generated[-3]:
                break
        
        return ' '.join(generated)
    
    def chat(self, user_message, max_response_tokens=30):
        """Chat interface"""
        # Simple prompt: just generate from last word
        words = user_message.strip().split()
        if not words:
            return "..."
        
        response = self.generate(words[-1], max_tokens=max_response_tokens)
        return response
    
    def close(self):
        self.conn.close()

def main():
    print("=" * 60)
    print("  HARTONOMOUS - PostgreSQL Geometric LLM")
    print("  Model: Qwen2.5-0.5B (167k atoms, 392M refs)")
    print("  Inference: Pure SQL spatial queries (no GPU)")
    print("=" * 60)
    print()
    
    llm = HartonomousLLM()
    
    if len(sys.argv) > 1:
        # Single query mode
        prompt = ' '.join(sys.argv[1:])
        print(f"Prompt: {prompt}")
        print(f"Response: {llm.generate(prompt, max_tokens=20)}")
    else:
        # Interactive mode
        print("Interactive mode. Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                response = llm.chat(user_input)
                print(f"Hartonomous: {response}\n")
                
            except KeyboardInterrupt:
                print("\n")
                break
            except Exception as e:
                print(f"Error: {e}\n")
    
    llm.close()
    print("\nGoodbye!")

if __name__ == '__main__':
    main()
