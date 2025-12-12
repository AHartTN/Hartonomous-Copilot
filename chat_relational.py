#!/usr/bin/env python3
"""
Hartonomous Chat - Using Relational Atoms (Weight Connections)
"""
import psycopg2
import sys
import re

class HartonomousRelationalLLM:
    def __init__(self):
        self.conn = psycopg2.connect(database='hartonomous', user='postgres', host='localhost', password='postgres')
        self.cur = self.conn.cursor()
        
    def predict_next_token(self, current_token_text):
        """Predict next token using relational atoms (weight connections)."""
        # Clean BPE encoding
        current_token_text = current_token_text.replace(' ', 'Ġ')
        
        query = """
            WITH current_tok AS (
                SELECT h_xy, h_yz, h_zm
                FROM atoms
                WHERE modality = 'compositional'
                AND meta->>'text' = %s
                LIMIT 1
            ),
            -- Find all relational atoms where source is current token
            weighted_connections AS (
                SELECT 
                    ST_M(ST_EndPoint(ST_GeometryN(r.geom, 1))) as target_hxy,
                    ST_M(ST_EndPoint(ST_GeometryN(r.geom, 2))) as target_hyz,
                    ST_M(ST_EndPoint(ST_GeometryN(r.geom, 3))) as target_hzm,
                    ST_Z(ST_EndPoint(ST_GeometryN(r.geom, 1))) as weight
                FROM atoms r
                CROSS JOIN current_tok c
                WHERE r.modality = 'relational'
                AND ST_M(ST_StartPoint(ST_GeometryN(r.geom, 1))) = c.h_xy
            )
            SELECT 
                a.meta->>'text' as token,
                wc.weight
            FROM weighted_connections wc
            JOIN atoms a ON (a.h_xy = wc.target_hxy AND a.h_yz = wc.target_hyz AND a.h_zm = wc.target_hzm)
            WHERE a.modality = 'compositional'
            ORDER BY wc.weight DESC
            LIMIT 5;
        """
        
        try:
            self.cur.execute(query, (current_token_text,))
            results = self.cur.fetchall()
            if results:
                # Return top prediction
                return results[0][0].replace('Ġ', ' ')
            return None
        except Exception as e:
            print(f"Query error: {e}")
            return None
    
    def generate(self, prompt, max_tokens=50):
        """Generate text by following relational atoms."""
        tokens = prompt.strip().split()
        generated = []
        
        current = tokens[-1] if tokens else None
        
        for _ in range(max_tokens):
            if not current:
                break
                
            next_token = self.predict_next_token(current)
            
            if not next_token:
                break
            
            generated.append(next_token)
            current = next_token.strip()
            
            # Stop on repetition
            if len(generated) > 2 and next_token == generated[-3]:
                break
        
        return ' '.join(generated)
    
    def close(self):
        self.conn.close()

def main():
    print("=" * 70)
    print("  HARTONOMOUS - Relational Atom Inference")
    print("  Using weight connections between tokens (112k relations)")
    print("=" * 70)
    print()
    
    llm = HartonomousRelationalLLM()
    
    if len(sys.argv) > 1:
        prompt = ' '.join(sys.argv[1:])
        print(f"Prompt: {prompt}")
        response = llm.generate(prompt, max_tokens=30)
        print(f"Response: {response}")
    else:
        print("Interactive mode. Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                response = llm.generate(user_input, max_tokens=30)
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
