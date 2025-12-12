#!/usr/bin/env python3
"""Final complete ingestion with all fixes"""
import struct, hashlib, subprocess, sys

def hilbert_encode(data_bytes):
    h = hashlib.sha256(data_bytes).digest()
    return (int.from_bytes(h[0:7], 'little') % (2**50),
            int.from_bytes(h[7:14], 'little') % (2**50),
            int.from_bytes(h[14:21], 'little') % (2**50))

def read_string(f):
    l = struct.unpack('<Q', f.read(8))[0]
    if l > 100000: f.seek(l, 1); return None
    return f.read(l).decode('utf-8', errors='replace')

print("=== FINAL Complete Qwen2.5 Ingestion ===\n")

# Parse vocabulary
tokens = []
with open('/tmp/model.gguf', 'rb') as f:
    f.read(12); kv_count = struct.unpack('<Q', f.read(8))[0]
    for i in range(kv_count):
        key = read_string(f); vtype = struct.unpack('<I', f.read(4))[0]
        if key == 'tokenizer.ggml.tokens' and vtype == 9:
            struct.unpack('<I', f.read(4)); alen = struct.unpack('<Q', f.read(8))[0]
            for j in range(alen):
                token = read_string(f)
                if token: tokens.append(token)
                if (j+1) % 50000 == 0: print(f"  {j+1}/{alen}")
            break
        else:
            if vtype == 8: read_string(f)
            elif vtype in [4,5,6]: f.read(4)
            elif vtype in [10,11,12]: f.read(8)
            elif vtype == 9:
                at = struct.unpack('<I', f.read(4))[0]; al = struct.unpack('<Q', f.read(8))[0]
                if at == 8:
                    for _ in range(al): read_string(f)
                elif at in [4,5,6]: f.seek(4*al, 1)
                else: f.seek(8*al, 1)

print(f"✓ {len(tokens)} tokens loaded")

# Collect unique bytes
all_bytes = set()
for token in tokens:
    for b in token.encode('utf-8'): all_bytes.add(b)

print(f"  Unique bytes: {len(all_bytes)}")
print(f"  Expected atoms: {len(all_bytes) + len(tokens):,}\n")

# Generate batches
byte_atoms = {}
batches = []
batch = "BEGIN;\n"; count = 0

for byte_val in sorted(all_bytes):
    vb = bytes([byte_val]); h = hilbert_encode(vb); byte_atoms[byte_val] = h
    ch = hashlib.sha256(vb).hexdigest()
    batch += f"INSERT INTO atoms (h_xy,h_yz,h_zm,geom,value,modality,content_hash,meta,created_by) VALUES ({h[0]},{h[1]},{h[2]},ST_GeomFromText('GEOMETRYCOLLECTION ZM(POINT ZM({h[0]} {h[1]} {h[2]} {byte_val}))',0),decode('{vb.hex()}','hex'),'discrete',decode('{ch}','hex'),'{{}}'::jsonb,'final_ingest') ON CONFLICT (h_xy,h_yz,h_zm) DO UPDATE SET ref_count=atoms.ref_count+1;\n"
    count += 1

print("Generating token SQL...")
for idx, token in enumerate(tokens):
    tb = token.encode('utf-8'); h = hilbert_encode(tb); ch = hashlib.sha256(tb).hexdigest()
    pts = [f"{byte_atoms[b][0]} {byte_atoms[b][1]} {byte_atoms[b][2]} {byte_atoms[b][0]}" for b in tb]
    geom = f"POINT ZM({pts[0]})" if len(pts)==1 else f"LINESTRING ZM({', '.join(pts)})"
    token_esc = token.replace('\\','\\\\').replace('"','\\"').replace('\n','\\n')[:100]
    batch += f"INSERT INTO atoms (h_xy,h_yz,h_zm,geom,value,modality,content_hash,meta,created_by) VALUES ({h[0]},{h[1]},{h[2]},ST_GeomFromText('GEOMETRYCOLLECTION ZM({geom})',0),NULL,'compositional',decode('{ch}','hex'),'{{"text":"{token_esc}"}}'::jsonb,'final_ingest') ON CONFLICT (h_xy,h_yz,h_zm) DO UPDATE SET ref_count=atoms.ref_count+1;\n"
    count += 1
    if count >= 1000:
        batch += "COMMIT;\n"; batches.append(batch); batch = "BEGIN;\n"; count = 0
    if (idx+1) % 50000 == 0: print(f"  {idx+1}/{len(tokens)}")

if count > 0: batch += "COMMIT;\n"; batches.append(batch)

print(f"✓ {len(batches)} batches generated\n")
print("Executing (may take 10-15 minutes)...")

for i, b in enumerate(batches):
    with open(f'/tmp/final_batch_{i}.sql', 'w') as f: f.write(b)
    subprocess.run(['sudo','-u','postgres','psql','-d','hartonomous','-f',f'/tmp/final_batch_{i}.sql'],
                   capture_output=True)
    if (i+1) % 20 == 0: print(f"  {i+1}/{len(batches)}")

print("\n✓ COMPLETE\n")
subprocess.run(['sudo','-u','postgres','psql','-d','hartonomous','-c',
                "SELECT COUNT(*) FILTER (WHERE modality='discrete' AND created_by='final_ingest') as bytes, COUNT(*) FILTER (WHERE modality='compositional' AND created_by='final_ingest') as tokens, SUM(ref_count) FILTER (WHERE created_by='final_ingest') as refs FROM atoms;"])
