# Hartonomous Build & Run Guide

## Prerequisites

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    zlib1g-dev \
    postgresql-16 \
    postgis \
    python3-dev

# macOS
brew install cmake eigen zlib postgresql@16 postgis

# Windows (use vcpkg)
vcpkg install eigen3 zlib
```

### Python Dependencies
```bash
pip install -r requirements.txt
# Or manually:
pip install numpy safetensors psycopg2-binary tqdm
```

## Build Steps

### 1. Build Native Library

```bash
cd native
mkdir -p build && cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON

# Build (parallel)
cmake --build . -j$(nproc)

# Test
ctest --output-on-failure

# Install
sudo cmake --install .
```

### 2. Setup Database

```bash
# Create database
createdb hartonomous

# Run schema
psql hartonomous < sql/schema_v2_manifold.sql
```

### 3. Install Python Package

```bash
# Development mode
pip install -e .

# Test native bindings
python -c "from hartonomous.native import version; print(version())"
```

## Running Ingestion

### Option 1: Using Safetensors Model

```bash
# Download model (if not already)
# Place in /data/models/qwen2.5-0.5b/

# Run ingestion
python -m hartonomous.ingest_manifold /data/models/qwen2.5-0.5b
```

### Option 2: Using GGUF Model

```bash
# Convert GGUF to SafeTensors first (TODO)
python scripts/gguf_to_safetensors.py models/qwen2.5-0.5b.gguf

# Then ingest
python -m hartonomous.ingest_manifold /data/models/qwen2.5-0.5b
```

## Verification

### Check Data

```sql
-- Connect
psql hartonomous

-- Count atoms
SELECT atom_type, COUNT(*), AVG(ref_count) as avg_refs
FROM atoms
GROUP BY atom_type;

-- Sample token atoms
SELECT
    encode(raw_value, 'escape') as token,
    h_xy, h_yz, h_zm, h_my,
    ref_count
FROM atoms
WHERE atom_type = 'token'
LIMIT 10;

-- Check semantic similarity (k-NN)
SELECT
    a2.raw_value::text as similar_token,
    ST_Distance(a1.geom, a2.geom) as distance
FROM atoms a1, atoms a2
WHERE a1.raw_value = 'hello'::bytea
  AND a1.atom_type = 'token'
  AND a2.atom_type = 'token'
  AND a2.h_xy != a1.h_xy
ORDER BY a1.geom <-> a2.geom
LIMIT 10;
```

### Performance Benchmarks

```bash
# Run benchmarks
cd native/build
./benchmarks/hilbert_benchmark
./benchmarks/manifold_benchmark
```

## Troubleshooting

### Native library not found
```bash
# Check LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Or copy to Python package
cp native/build/libhartonomous_native.so src/hartonomous/
```

### Database connection failed
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
psql hartonomous -c "SELECT version()"
```

### Out of memory during ingestion
```bash
# Reduce batch size in ingest_manifold.py
# Or increase PostgreSQL memory:
# Edit /etc/postgresql/16/main/postgresql.conf
shared_buffers = 4GB
work_mem = 256MB
```

## Next Steps

After successful ingestion:

1. **Test semantic queries** (see `examples/semantic_search.py`)
2. **Implement inference** (see `src/hartonomous/inference/`)
3. **Build API** (see `Hartonomous.API/`)
4. **Deploy** (see `docker-compose.yml`)

## Architecture Summary

```
Application (Python/C#)
    ↓ ctypes/P/Invoke
Native Library (C++20)
    - Hilbert encoding (Skilling's algorithm)
    - Laplacian Eigenmaps
    - PCA projection
    ↓
PostgreSQL + PostGIS
    - atoms (4-manifold Hilbert PK)
    - relations (meta-learning stats)
    - GIST spatial index
```

The **embedding vectors from training ARE the semantic manifold**.
Spatial proximity = Semantic similarity (guaranteed by manifold learning).
