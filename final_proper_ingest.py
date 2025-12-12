#!/usr/bin/env python3
import struct, hashlib, subprocess, json

def hilbert_encode(data):
    h = hashlib.sha256(data).digest()
    return (int.from_bytes(h[0:7], 'little') % (2**50),
            int.from_bytes(h[7:14], 'little') % (2**50),
            int.from_bytes(h[14:21], 'little') % (2**50))

def read_string(f):
    l = struct.unpack('<Q', f.read(8))[0]
    if l > 100000: f.seek(l, 1); return None
    return f.read(l).decode('utf-8', errors='replace')

print("=== FINAL PROPER Qwen2.5-0.5B Ingestion ===\n")

tokens = []
with open('/tmp/model.gguf', 'rb') as f:
    f.read(12); kv = struct.unpack('<Q', f.read(8))[0]
    for i in range(kv):
        k = read_string(f); vt = struct.unpack('<I', f.read(4))[0]
        if k == 'tokenizer.ggml.tokens' and vt == 9:
            struct.unpack('<I', f.read(4)); al = struct.unpack('<Q', f.read(8))[0]
            for j in range(al):
                t = read_string(f)
                if t: tokens.append(t)
                if (j+1) % 50000 == 0: print(f"  {j+1}/{al}")
            break
        else:
            if vt == 8: read_string(f)
            elif vt in [4,5,6]: f.read(4)
            elif vt in [10,11,12]: f.read(8)
            elif vt == 9:
                at = struct.unpack('<I', f.read(4))[0]; al = struct.unpack('<Q', f.read(8))[0]
                if at == 8:
                    for _ in range(al): read_string(f)
                elif at in [4,5,6]: f.seek(4*al, 1)
                else: f.seek(8*al, 1)

print(f"✓ {len(tokens)} tokens\n")

unique_bytes = set()
single = multi = 0
for token in tokens:
    tb = token.encode('utf-8')
    for b in tb: unique_bytes.add(b)
    if len(tb) == 1: single += 1
    else: multi += 1

print(f"Single-byte: {single}, Multi-byte: {multi}")
print(f"Expected: {len(unique_bytes)} bytes + {multi} compositions = {len(unique_bytes)+multi} atoms\n")

byte_atoms = {}
sql = "BEGIN;\n"

for bv in sorted(unique_bytes):
    vb = bytes([bv]); h = hilbert_encode(vb); byte_atoms[bv] = h
    ch = hashlib.sha256(vb).hexdigest()
    sql += f"INSERT INTO atoms (h_xy,h_yz,h_zm,geom,value,modality,content_hash,meta,created_by) VALUES ({h[0]},{h[1]},{h[2]},ST_GeomFromText('GEOMETRYCOLLECTION ZM(POINT ZM({h[0]} {h[1]} {h[2]} {bv}))',0),decode('{vb.hex()}','hex'),'discrete',decode('{ch}','hex'),'{{}}'::jsonb,'final') ON CONFLICT (h_xy,h_yz,h_zm) DO UPDATE SET ref_count=atoms.ref_count+1;\n"

for idx, token in enumerate(tokens):
    tb = token.encode('utf-8')
    if len(tb) > 1:
        h = hilbert_encode(tb); ch = hashlib.sha256(tb).hexdigest()
        pts = [f"{byte_atoms[b][0]} {byte_atoms[b][1]} {byte_atoms[b][2]} {byte_atoms[b][0]}" for b in tb]
        geom = f"LINESTRING ZM({', '.join(pts)})"
        meta = json.dumps({"text": token[:100]})
        sql += f"INSERT INTO atoms (h_xy,h_yz,h_zm,geom,value,modality,content_hash,meta,created_by) VALUES ({h[0]},{h[1]},{h[2]},ST_GeomFromText('GEOMETRYCOLLECTION ZM({geom})',0),NULL,'compositional',decode('{ch}','hex'),'{meta}'::jsonb,'final') ON CONFLICT (h_xy,h_yz,h_zm) DO UPDATE SET ref_count=atoms.ref_count+1;\n"
    if (idx+1) % 50000 == 0: print(f"  {idx+1}/{len(tokens)}")

sql += "COMMIT;\n"

with open('/tmp/final.sql', 'w') as f: f.write(sql)
print("\nExecuting...")
subprocess.run(['sudo','-u','postgres','psql','-d','hartonomous','-f','/tmp/final.sql'], timeout=300)
print("\n✓ DONE\n")
subprocess.run(['sudo','-u','postgres','psql','-d','hartonomous','-c',
                "SELECT COUNT(*) FILTER (WHERE modality='discrete' AND created_by='final') as bytes, COUNT(*) FILTER (WHERE modality='compositional' AND created_by='final') as comps, SUM(ref_count) FILTER (WHERE created_by='final') as refs FROM atoms;"])
