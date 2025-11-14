import os

from pgvector.psycopg2 import register_vector
import psycopg2
from psycopg2.extras import RealDictCursor

from src.config import POSTGRES_DB_URL


class Manager:

    def __init__(self, url=None):
        self.db_url = url if url else POSTGRES_DB_URL
        self.cursor = None
        self.conn = None

    def _connect(self):
        self.conn = psycopg2.connect(dsn=self.db_url)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.cursor.execute("SET client_min_messages TO error;")
        self.conn.commit()
        register_vector(self.conn)

    def _disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._disconnect()
