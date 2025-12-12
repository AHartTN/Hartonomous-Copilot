#!/usr/bin/env python3
import psycopg2

def chat(prompt):
    conn = psycopg2.connect(dbname='hartonomous')
    cur = conn.cursor()
    
    words = prompt.split()
    response = []
    
    for word in words:
        cur.execute("SELECT id FROM atoms WHERE raw_value = %s LIMIT 1", (word.encode('utf-8'),))
        result = cur.fetchone()
        
        if result:
            atom_id = result[0]
            # Find nearest atom in geometric space
            cur.execute("""
                SELECT raw_value FROM atoms a2
                WHERE a2.id != %s AND length(a2.raw_value) > 0
                ORDER BY (SELECT geom FROM atoms WHERE id = %s) <-> a2.geom
                LIMIT 1
            """, (atom_id, atom_id))
            nearest = cur.fetchone()
            if nearest:
                response.append(nearest[0].decode('utf-8', errors='ignore'))
        else:
            response.append(word)
    
    conn.close()
    return ' '.join(response)

print("HARTONOMOUS Chat")
print("="*50)
while True:
    try:
        prompt = input("\nYou: ")
        if prompt.lower() in ['quit', 'exit']:
            break
        print(f"AI:  {chat(prompt)}")
    except (KeyboardInterrupt, EOFError):
        break
print("\nGoodbye!")
