#!/usr/bin/env python3
"""
CLI command to ingest a model into the database
"""
import sys
from pathlib import Path
from ..db import Database
from ..ingestion.model import ModelIngester


def ingest_command(model_path: str, vocab_only: bool = False, embeddings_only: bool = False):
    """Ingest model from directory"""
    model_dir = Path(model_path)
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return 1
    
    # Check required files
    required = ['vocab.json', 'model.safetensors']
    for req in required:
        if not (model_dir / req).exists():
            print(f"❌ Required file not found: {req}")
            return 1
    
    print(f"✓ Model directory: {model_dir}")
    
    # Connect to database
    with Database() as db:
        ingester = ModelIngester(db, model_dir)
        
        if vocab_only:
            ingester.ingest_vocabulary()
        elif embeddings_only:
            ingester.ingest_embeddings()
        else:
            ingester.run_full_ingestion()
        
        ingester.print_stats()
    
    return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m hartonomous.cli.ingest <model_dir> [--vocab-only|--embeddings-only]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    vocab_only = '--vocab-only' in sys.argv
    embeddings_only = '--embeddings-only' in sys.argv
    
    sys.exit(ingest_command(model_path, vocab_only, embeddings_only))
