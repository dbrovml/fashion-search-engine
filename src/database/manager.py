"""Context manager for PostgreSQL database connection."""

from types import TracebackType
from typing import Optional

from pgvector.psycopg2 import register_vector
import psycopg2
from psycopg2.extras import RealDictCursor

from src.config import POSTGRES_DB_URL


class Manager:

    def __init__(self, url: Optional[str] = None) -> None:
        """Configure the manager with a database URL."""
        self.db_url = url if url else POSTGRES_DB_URL
        self.cursor = None
        self.conn = None

    def _connect(self) -> None:
        """Establish the connection and ensure extensions."""
        self.conn = psycopg2.connect(dsn=self.db_url)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.cursor.execute("SET client_min_messages TO error;")
        self.conn.commit()
        register_vector(self.conn)

    def _disconnect(self) -> None:
        """Close cursor and connection if present."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def __enter__(self) -> "Manager":
        """Enter the context and return the manager."""
        self._connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Disconnect regardless of context outcome."""
        self._disconnect()
