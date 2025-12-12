#!/usr/bin/env python3
"""
Hartonomous CLI - Main entry point

Usage:
    hartonomous status          # Show database statistics
    hartonomous ingest <model>  # Ingest a model
    hartonomous chat            # Interactive chat
"""

import sys
import argparse
from ..db import Database

def cmd_status(args):
    """Show database status"""
    with Database() as db:
        stats = db.get_stats()
        
        print("=" * 60)
        print("HARTONOMOUS DATABASE STATUS")
        print("=" * 60)
        print(f"Atoms:              {stats['atoms']:>15,}")
        print(f"Compositions:       {stats['compositions']:>15,}")
        print(f"Token Embeddings:   {stats['token_embeddings']:>15,}")
        print()
        print("Relations by type:")
        for rel_type, count in stats['relations'].items():
            print(f"  {rel_type:20s} {count:>10,}")
        print("=" * 60)

def cmd_ingest(args):
    """Ingest a model"""
    print(f"TODO: Ingest {args.model}")

def cmd_chat(args):
    """Interactive chat"""
    print("TODO: Chat interface")

def main():
    parser = argparse.ArgumentParser(description='Hartonomous CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # status command
    parser_status = subparsers.add_parser('status', help='Show database statistics')
    parser_status.set_defaults(func=cmd_status)
    
    # ingest command
    parser_ingest = subparsers.add_parser('ingest', help='Ingest a model')
    parser_ingest.add_argument('model', help='Model name or path')
    parser_ingest.set_defaults(func=cmd_ingest)
    
    # chat command
    parser_chat = subparsers.add_parser('chat', help='Interactive chat')
    parser_chat.set_defaults(func=cmd_chat)
    
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    args.func(args)

if __name__ == '__main__':
    main()
