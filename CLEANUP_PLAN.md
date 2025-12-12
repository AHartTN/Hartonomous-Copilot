# Hartonomous Cleanup & Consolidation Plan

**Date**: 2025-12-12
**Current State**: MESS - 17 scattered Python scripts, incomplete ingestion, broken chat interfaces

---

## CURRENT INVENTORY

### Database (hartonomous)
- **142,145 atoms** - byte primitives (0-9, etc.) with Hilbert IDs
- **1,241,536 compositions** - parent→child relations showing token→byte breakdown
- **48,938 relations** - all type='composition', no layer_name
- **40,029 token_embeddings** - 40K/152K tokens have embedding_atom_id arrays
- **37 connections** - test/experimental data
- **0 weight_matrices** - EMPTY
- **2 custom functions**: decrement_ref_count(), dot_product()

### Python Scripts (ROOT - 17 files, 2,477 total lines)
1. **ingest_unified_graph.py** (395 lines) - GGUF parser, attempts full model ingest
2. **chat_working.py** (214 lines) - Tries to use compositions for inference (BROKEN - wrong column names)
3. **ingest_atomic.py** (195 lines) - Atomic decomposition approach
4. **demo_final.py** (187 lines) - Demo script
5. **build_connections.py** (176 lines) - Build connection graph
6. **ingest_complete.py** (149 lines) - "Complete unified model ingestion"
7. **chat_relational.py** (127 lines) - Relational chat
8. **ingest_model.py** (124 lines) - Model ingestion
9. **chat.py** (119 lines) - Basic chat
10. **ingest_weights.py** (117 lines) - Weight ingestion
11. **simple_chat.py** (108 lines) - Simple chat interface
12. **inference.py** (102 lines) - Inference module
13. **inference_demo.py** (98 lines) - Inference demo
14. **complete_model_ingest_fixed.py** (88 lines) - "Final complete ingestion with all fixes"
15. **final_proper_ingest.py** (78 lines) - Another "final" version
16. **download_and_ingest.py** (70 lines) - Download and ingest
17. **chat_geo.py** (43 lines) - Geometric chat

### Python Scripts (src/ directory)
- `src/inference_engine.py` (255 lines) - Inference engine with PL/Python
- `src/ingest_full_model.py` (176 lines) - GGUF parsing

### Scripts (scripts/ directory)
- `scripts/ingest_unified_graph.py` (308 lines) - Another unified graph ingester
- `scripts/ingest_unified_graph_impl.py` (186 lines) - Implementation

### C# Projects (incomplete, all modified in git)
- Hartonomous.Core/ - Domain models, Hilbert encoding
- Hartonomous.Infrastructure/ - EF Core (empty)
- Hartonomous.Api/ - (empty)
- Hartonomous.Cli/ - CLI app (incomplete)

### Models Available
- `/data/models/qwen2.5-0.5b/` - 988MB SafeTensors + vocab (151,936 tokens)
- `/home/ahart/Projects/Hartonomous-Copilot/models/qwen2.5-0.5b.gguf` - 380MB
- Ollama models: all-minilm (45MB), llama3.2:1b (1.3GB), tinyllama (637MB), qwen2.5:0.5b (397MB)

### SQL Schema
- `sql/001_initial_schema.sql` (393 lines) - Partitioned atoms table, functions, triggers
- `sql/002_graph_schema.sql` (44 lines) - Relations, token_embeddings, views

---

## PROBLEMS

### 1. **Script Chaos**
- 17+ Python scripts with overlapping functionality
- No clear "canonical" version
- Multiple attempts at same task (ingest_complete, complete_model_ingest_fixed, final_proper_ingest)
- Broken references (chat_working.py uses `atom_id` column that doesn't exist)

### 2. **Incomplete Ingestion**
- Only 40K/152K tokens have embeddings
- NO transformer weights ingested (Q/K/V projections, FFN layers)
- NO attention mechanism
- NO actual inference possible

### 3. **No Working CLI**
- All chat scripts are broken or use placeholder logic
- No unified entry point
- No way to actually interact with the system

### 4. **Mixed Architectures**
- C# project started but abandoned
- Python scripts scattered
- No clear direction

### 5. **No Cleanup**
- No .gitignore
- __pycache__ committed
- Build artifacts committed
- models/ directory (380MB) not in .gitignore

---

## SOLUTION: CONSOLIDATE INTO CANONICAL STRUCTURE

### Phase 1: Immediate Cleanup (30 min)
1. Create `.gitignore`
2. Remove __pycache__, build artifacts
3. Archive all 17 root Python scripts to `archive/`
4. Create clean `src/hartonomous/` package structure

### Phase 2: Build Canonical Tools (2-3 hours)
Create ONE set of working tools:

```
src/hartonomous/
├── __init__.py
├── db.py                   # Database connection
├── hilbert.py              # Hilbert encoding (pure Python, no C# dependency)
├── ingestion/
│   ├── __init__.py
│   ├── tokenizer.py        # Tokenizer ingestion (vocab + metadata)
│   ├── embeddings.py       # Embedding matrix ingestion
│   ├── weights.py          # Transformer weight ingestion (Q/K/V, FFN)
│   └── pipeline.py         # Full model ingestion pipeline
├── inference/
│   ├── __init__.py
│   ├── attention.py        # Multi-head attention (SQL + PL/Python)
│   ├── ffn.py              # Feed-forward network
│   └── generate.py         # Text generation
└── cli/
    ├── __init__.py
    ├── ingest.py           # CLI: hartonomous ingest <model>
    ├── chat.py             # CLI: hartonomous chat
    └── query.py            # CLI: hartonomous query <sql>
```

### Phase 3: Complete Ingestion (1-2 hours)
- Ingest ALL 152K token embeddings
- Ingest transformer weights for layer 0 (prove it works)
- Test reconstruction

### Phase 4: Working Inference (2-3 hours)
- Implement attention in PL/Python
- Implement FFN
- Test single-layer forward pass
- Build text generation loop

### Phase 5: CLI Interface (1 hour)
- `hartonomous ingest qwen2.5:0.5b` - ingest model
- `hartonomous chat` - interactive chat
- `hartonomous status` - show database stats

---

## EXECUTION PLAN

### Step 1: Create .gitignore
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.dylib
*.dll
*.egg-info/
dist/
build/
*.log
models/*.gguf
models/*.safetensors
src/**/__pycache__/
*.swp
*.swo
*~
.DS_Store
obj/
bin/
```

### Step 2: Archive Old Scripts
```bash
mkdir -p archive/attempts
mv *.py archive/attempts/
git add archive/
```

### Step 3: Build Clean Package
Start with minimal working modules, test each one.

### Step 4: Test & Document
- Write tests for each module
- Update README with actual working commands
- Create ARCHITECTURE.md that reflects reality

---

## SUCCESS CRITERIA

At the end, you should be able to run:

```bash
# Install
pip install -e .

# Ingest a model
hartonomous ingest --model qwen2.5:0.5b --source ollama

# Chat
hartonomous chat
> Hello, how are you?
< I'm doing well, thank you for asking!

# Query database
hartonomous status
Atoms: 500,000
Relations: 2,000,000
Tokens: 151,936
Embeddings: 151,936 (100%)
Weights: 24 layers

# Explain a prediction
hartonomous explain "why did you say 'well'?"
Token 'well' selected via:
  1. Embedding similarity to 'how' (0.87)
  2. Attention weight from position 2 (0.92)
  3. FFN output probability (0.68)
```

---

## DECISION: Python or C#?

**Recommendation: PYTHON**
- All current work is in Python
- Database operations easier in Python (psycopg2)
- NumPy for tensor operations
- Faster iteration
- Can always add C# API layer later

Abandon C# projects for now, focus on working Python implementation.

---

## NEXT ACTIONS

1. Create .gitignore
2. Archive old scripts
3. Build src/hartonomous/ package structure
4. Implement hilbert.py (pure Python)
5. Implement db.py (connection manager)
6. Implement ingestion/tokenizer.py
7. Test with qwen2.5:0.5b vocabulary
8. Continue through pipeline...

