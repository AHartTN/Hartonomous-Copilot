#!/usr/bin/env python3
"""
GGUF Model Ingestion - Extract weights and store as geometric atoms
"""
import struct
import sys
import psycopg2
import hashlib
from pathlib import Path

def read_gguf_header(file_path):
    """Parse GGUF file header"""
    with open(file_path, 'rb') as f:
        # GGUF magic number
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Not a GGUF file: {magic}")
        
        # Version
        version = struct.unpack('<I', f.read(4))[0]
        print(f"GGUF version: {version}")
        
        # Tensor count
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        print(f"Tensors: {tensor_count}")
        
        # Metadata KV count  
        kv_count = struct.unpack('<Q', f.read(8))[0]
        print(f"Metadata entries: {kv_count}")
        
        return version, tensor_count, kv_count

def extract_weights_sample(file_path, limit=1000):
    """Extract first N weight values from GGUF for testing"""
    weights = []
    with open(file_path, 'rb') as f:
        # Skip to data section (simplified - real parser needs full header parse)
        f.seek(1024)  # Rough estimate
        
        # Read float16 values
        for i in range(limit):
            try:
                bytes_val = f.read(2)
                if len(bytes_val) < 2:
                    break
                # Store raw bytes
                weights.append(bytes_val)
            except:
                break
    
    return weights

def ingest_weight_atom(conn, weight_bytes, index):
    """Insert single weight value as atom"""
    # Generate SDI hash (simplified)
    sdi = hashlib.sha256(weight_bytes + index.to_bytes(8, 'little')).digest()
    
    # For now, use simple Hilbert encoding (would use proper implementation)
    h_xy = int.from_bytes(sdi[0:8], 'little') % (2**56)
    h_yz = int.from_bytes(sdi[8:16], 'little') % (2**56)
    h_zm = int.from_bytes(sdi[16:24], 'little') % (2**56)
    
    # Create POINT geometry (WKT format)
    geom_wkt = f'POINT({h_xy} {h_yz} {h_zm})'
    
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO atoms (h_xy, h_yz, h_zm, geom, value, modality, content_hash, created_by)
            VALUES (%s, %s, %s, ST_GeomFromText(%s, 0), %s, 'continuous', %s, 'ingestion_script')
            ON CONFLICT (h_xy, h_yz, h_zm) DO UPDATE SET ref_count = atoms.ref_count + 1
        """, (h_xy, h_yz, h_zm, geom_wkt, weight_bytes, sdi))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error inserting atom {index}: {e}")
        return False

def main():
    import sys
    model_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/model.gguf"
    
    print(f"Ingesting: {model_file}")
    print("="*60)
    
    # Parse header
    version, tensor_count, kv_count = read_gguf_header(model_file)
    
    # Extract sample weights
    print("\nExtracting weight samples...")
    weights = extract_weights_sample(model_file)
    print(f"Extracted {len(weights)} weight values")
    
    # Connect to database
    print("\nConnecting to database...")
    conn = psycopg2.connect(
        host="localhost",
        database="hartonomous",
        user="postgres",
        password=""
    )
    
    # Ingest atoms
    print("\nIngesting atoms...")
    success_count = 0
    for i, weight in enumerate(weights):
        if ingest_weight_atom(conn, weight, i):
            success_count += 1
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(weights)}")
    
    print(f"\n✓ Successfully ingested {success_count}/{len(weights)} atoms")
    
    # Query results
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM atoms WHERE modality = 'continuous'")
    total = cur.fetchone()[0]
    print(f"✓ Total continuous atoms in database: {total}")
    
    conn.close()

if __name__ == "__main__":
    main()
