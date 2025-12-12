"""Database connection and query utilities"""
import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
from contextlib import contextmanager
from typing import Optional, List, Tuple, Any

class Database:
    """PostgreSQL connection manager for Hartonomous"""
    
    def __init__(self, 
                 dbname: str = 'hartonomous',
                 user: str = 'postgres',
                 password: str = 'postgres',
                 host: str = 'localhost',
                 port: int = 5432):
        self.conn_params = {
            'dbname': dbname,
            'user': user,
            'password': password,
            'host': host,
            'port': port
        }
        self.conn = None
        self.cur = None
    
    def connect(self):
        """Establish database connection"""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(**self.conn_params)
            self.cur = self.conn.cursor()
        return self
    
    def close(self):
        """Close database connection"""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        self.conn = None
        self.cur = None
    
    def execute(self, sql: str, params: Optional[Tuple] = None) -> 'Database':
        """Execute a query"""
        if self.cur is None:
            self.connect()
        self.cur.execute(sql, params)
        return self
    
    def fetchone(self) -> Optional[Tuple]:
        """Fetch one result"""
        return self.cur.fetchone() if self.cur else None
    
    def fetchall(self) -> List[Tuple]:
        """Fetch all results"""
        return self.cur.fetchall() if self.cur else []
    
    def commit(self):
        """Commit transaction"""
        if self.conn:
            self.conn.commit()
    
    def rollback(self):
        """Rollback transaction"""
        if self.conn:
            self.conn.rollback()
    
    def batch_insert(self, sql: str, data: List[Tuple], page_size: int = 1000):
        """Batch insert with execute_batch"""
        if self.cur is None:
            self.connect()
        execute_batch(self.cur, sql, data, page_size=page_size)
    
    def get_stats(self) -> dict:
        """Get database statistics"""
        stats = {}
        
        # Atom count
        self.execute("SELECT COUNT(*) FROM atoms")
        stats['atoms'] = self.fetchone()[0]
        
        # Relations count
        self.execute("SELECT relation_type, COUNT(*) FROM relations GROUP BY relation_type")
        stats['relations'] = {row[0]: row[1] for row in self.fetchall()}
        
        # Token embeddings
        self.execute("SELECT COUNT(*) FROM token_embeddings")
        stats['token_embeddings'] = self.fetchone()[0]
        
        # Compositions
        self.execute("SELECT COUNT(*) FROM compositions")
        stats['compositions'] = self.fetchone()[0]
        
        return stats
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        self.close()
