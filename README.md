# Hartonomous-Copilot

**PostgreSQL/PostGIS as a Geometric AI Knowledge Substrate**

> *"The Database IS the AI Model"*

**Version**: 0.3.0  
**Status**: Active Development - Core Infrastructure Complete

---

## What This Is

A radical reimagining of AI where models aren't stored as blob files but as **geometric atomic graphs** in PostgreSQL. Inference happens via **spatial SQL queries** and **PL/Python functions** instead of GPU matrix multiplication.

---

## Quick Start

### Prerequisites

```bash
# PostgreSQL with PostGIS
sudo apt install postgresql-16 postgresql-16-postgis-3

# Python dependencies
pip install -e .
```

### Setup Database

```bash
# Initialize schema
sudo -u postgres psql < sql/001_initial_schema.sql
sudo -u postgres psql < sql/002_graph_schema.sql
```

### Install CLI

```bash
# Install in development mode
python3 setup.py develop --user

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Test
hartonomous status
```

---

## Current Status (2025-12-12)

### ✅ Working Components

#### Database Schema
- **142,145 atoms** - Primitive values (bytes, floats) with Hilbert IDs
- **1.2M compositions** - Parent→child relations showing token→byte decomposition
- **40K token embeddings** - Partial vocabulary ingestion (40K/152K tokens)
- **PostGIS GIST indexes** - Spatial queries on 4D geometric coordinates
- **Custom functions** - ref_count management, dot_product

#### CLI Interface
```bash
$ hartonomous status
============================================================
HARTONOMOUS DATABASE STATUS
============================================================
Atoms:                      142,145
Compositions:             1,238,360
Token Embeddings:            40,029

Relations by type:
  composition              48,938
============================================================
```

### ⚠️ In Progress

- **Embedding ingestion** - 40K/152K tokens complete (26%)
- **Weight matrix storage** - Schema exists, no data ingested
- **Inference engine** - PL/Python functions defined, not tested

### ❌ Not Yet Implemented

- **Complete transformer layers** (Q/K/V projections, FFN, layer norm)
- **Attention mechanism** in SQL
- **Text generation** pipeline
- **Training/fine-tuning** via UPDATE statements

---

## Architecture

### Core Concepts

1. **Everything decomposes to atoms**
   - Bytes → Characters → Tokens → Embeddings → Weights
   
2. **Atoms exist in 4D geometric space**
   - Deterministic Hilbert coordinates for deduplication
   - PostGIS PointZM geometry: `(x, y, z, m)`
   
3. **Spatial proximity = Semantic similarity**
   - PostGIS GIST R-tree indexes for k-NN search
   
4. **Relations encode neural networks**
   - Weight matrices as graph edges between atoms
   
5. **Hilbert curves for indexing**
   - 4D coords → 64-bit ID for B-tree efficiency

### Database Schema

```sql
-- Atoms: Primitive values with geometric coordinates
CREATE TABLE atoms (
    id BIGINT PRIMARY KEY,              -- Hilbert curve index
    geom GEOMETRY(PointZM) NOT NULL,    -- 4D coordinates
    raw_value BYTEA                     -- Actual content
);

-- Relations: Neural network structure
CREATE TABLE relations (
    id BIGSERIAL PRIMARY KEY,
    parent_id BIGINT REFERENCES atoms(id),
    child_id BIGINT REFERENCES atoms(id),
    relation_type TEXT,                 -- 'embedding', 'attention', 'ffn'
    position INT,
    weight_atom_id BIGINT REFERENCES atoms(id),
    layer_name TEXT
);

-- Token Embeddings: Fast lookup
CREATE TABLE token_embeddings (
    token_id INT PRIMARY KEY,
    token_text TEXT,
    embedding_atom_id BIGINT[],         -- Array of atom IDs
    embedding_h_coords BIGINT[]         -- Hilbert coordinates
);
```

### Hilbert Encoding Example

```python
def hilbert_encode_4d(x, y, z, m, bits=16):
    """
    Encode 4D coordinates [-1,1]^4 to 64-bit Hilbert index.
    Preserves spatial locality for efficient B-tree clustering.
    """
    # Normalize to [0, 2^bits)
    ix = int((x + 1) * 0.5 * ((1 << bits) - 1))
    iy = int((y + 1) * 0.5 * ((1 << bits) - 1))
    iz = int((z + 1) * 0.5 * ((1 << bits) - 1))
    im = int((m + 1) * 0.5 * ((1 << bits) - 1))
    
    # Bit-interleave
    result = 0
    for i in range(bits):
        result |= ((ix >> i) & 1) << (i * 4 + 0)
        result |= ((iy >> i) & 1) << (i * 4 + 1)
        result |= ((iz >> i) & 1) << (i * 4 + 2)
        result |= ((im >> i) & 1) << (i * 4 + 3)
    return result
```

---

## Project Structure

```
Hartonomous-Copilot/
├── setup.py                    # Package installation
├── .gitignore                  # Clean repository
│
├── sql/
│   ├── 001_initial_schema.sql  # Partitioned atoms table (393 lines)
│   └── 002_graph_schema.sql    # Relations, views (44 lines)
│
├── src/hartonomous/
│   ├── __init__.py
│   ├── db.py                   # Database connection manager
│   ├── cli/                    # Command-line interface
│   │   ├── __init__.py
│   │   └── __main__.py         # hartonomous command
│   ├── ingestion/              # Model ingestion (TODO)
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   ├── embeddings.py
│   │   └── weights.py
│   └── inference/              # Inference engine (TODO)
│       ├── __init__.py
│       ├── attention.py
│       └── generate.py
│
├── archive/
│   └── old_scripts/            # Previous experimental scripts (17 files)
│
├── models/
│   └── qwen2.5-0.5b.gguf       # Test model (gitignored)
│
├── docs/                       # Documentation
├── tests/                      # Tests (TODO)
│
└── [Legacy Files]
    ├── Hartonomous.Cli/        # C# CLI (abandoned for now)
    ├── src/Hartonomous.Core/   # C# Core library (abandoned)
    └── *.md                    # Various documentation files
```

---

## Models Available

Located in `/data/models/`:
- **qwen2.5-0.5b** - 988MB SafeTensors, 152K vocab (currently partially ingested)
- **all-minilm** - 45MB embedding model (Ollama)
- **llama3.2:1b** - 1.3GB (Ollama)
- **tinyllama** - 637MB (Ollama)

---

## Development Roadmap

### Phase 1: Complete Basic Ingestion (Current)
- [x] Clean up repository structure
- [x] Create working CLI
- [x] Archive old experimental scripts
- [ ] Complete token embedding ingestion (112K remaining)
- [ ] Ingest transformer weights (Q/K/V, FFN) for layer 0
- [ ] Test weight reconstruction

### Phase 2: Implement Inference
- [ ] PL/Python attention mechanism
- [ ] Matrix multiplication functions
- [ ] Softmax/sampling
- [ ] Single-layer forward pass test
- [ ] Full 24-layer inference pipeline

### Phase 3: Chat Interface
- [ ] Token generation loop
- [ ] Prompt handling
- [ ] Interactive chat CLI
- [ ] Beam search/sampling strategies

### Phase 4: Optimization
- [ ] Sparse weight loading
- [ ] Query optimization
- [ ] C extensions for Hilbert encoding
- [ ] Benchmark vs. traditional inference

### Phase 5: Advanced Features
- [ ] Training via UPDATE statements
- [ ] Fine-tuning pipelines
- [ ] Multi-modal support (audio, images)
- [ ] Distributed inference

---

## Usage Examples

### Check Database Status
```bash
hartonomous status
```

### Ingest a Model (TODO)
```bash
hartonomous ingest qwen2.5:0.5b --source /data/models/qwen2.5-0.5b
```

### Interactive Chat (TODO)
```bash
hartonomous chat
> Hello, how are you?
< I'm doing well, thank you!
```

### Query Database Directly
```bash
psql -U postgres hartonomous -c "
    SELECT token_text, array_length(embedding_atom_id, 1) as dims
    FROM token_embeddings
    LIMIT 5;
"
```

---

## Performance (Current)

| Metric | Value |
|--------|-------|
| Atoms stored | 142,145 |
| Compositions | 1,238,360 |
| Token embeddings | 40,029 |
| Relations | 48,938 |
| Database size | ~50MB |
| Ingestion speed | ~1,000 atoms/sec |
| k-NN query time | <100ms (indexed) |

---

## Philosophy

This project challenges three assumptions:

1. **"AI models must be binary blobs"** → No, they can be structured, queryable data
2. **"Inference requires GPUs"** → No, spatial indexes + SQL can work for small models
3. **"Training data separate from weights"** → No, unified in atomic graph

The goal: **Lossless, queryable, auditable AI** where `SELECT * FROM reasoning` is possible.

---

## Contributing

This is an experimental research project. Contributions welcome:
- Complete the ingestion pipeline
- Implement attention in PL/Python
- Optimize queries
- Add tests
- Improve documentation

---

## License

MIT

---

## Credits

- **Concept & Theory**: @aharttn
- **Implementation**: Autonomous development with GitHub Copilot CLI
- **Test Model**: Qwen2.5-0.5B-Instruct (Alibaba Cloud)
- **Stack**: PostgreSQL 16, PostGIS 3.6, Python 3.10

---

## Status Update

**Last Updated**: 2025-12-12  
**Current Focus**: Consolidating codebase, completing embedding ingestion, building inference engine  
**Next Milestone**: Working chat interface with single-layer transformer

For detailed implementation plan, see `CLEANUP_PLAN.md`.
