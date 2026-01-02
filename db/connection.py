"""
Database connection management for ollama-llm-proxy.

Handles SQLite (default) and PostgreSQL connections.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _ensure_data_dir() -> Path:
    """Ensure the data directory exists and return its path."""
    # Use /data in Docker (separate volume), otherwise relative to this file
    if Path("/app").exists():
        # Running in Docker container - use /data which is a mounted volume
        data_dir = Path("/data")
    else:
        # Running locally
        data_dir = Path(__file__).parent / "data"

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        # Test that we can write to it
        test_file = data_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        logger.info(f"Data directory ready: {data_dir}")
    except PermissionError as e:
        logger.error(f"Cannot write to data directory {data_dir}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error setting up data directory {data_dir}: {e}")
        raise

    return data_dir


def get_database_url() -> str:
    """
    Get database URL from environment or use default SQLite.

    Supports:
    - DATABASE_URL env var (for PostgreSQL or custom SQLite path)
    - Default: sqlite:////data/proxy.db (Docker) or local db/data/proxy.db
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        # Handle Heroku-style postgres:// URLs
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    data_dir = _ensure_data_dir()
    return f"sqlite:///{data_dir}/proxy.db"


def get_engine() -> Engine:
    """Get or create the database engine."""
    global _engine

    if _engine is None:
        url = get_database_url()
        logger.info(
            f"Connecting to database: {url.split('@')[-1] if '@' in url else url}"
        )

        # SQLite-specific settings
        if url.startswith("sqlite"):
            _engine = create_engine(
                url,
                connect_args={
                    "check_same_thread": False
                },  # Allow multi-threaded access
                echo=os.environ.get("SQL_DEBUG", "").lower() == "true",
            )

            # Enable foreign keys for SQLite
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            # PostgreSQL or other databases
            _engine = create_engine(
                url,
                echo=os.environ.get("SQL_DEBUG", "").lower() == "true",
                pool_pre_ping=True,  # Verify connections before use
            )

    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the session factory."""
    global _SessionLocal

    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )

    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.

    Usage:
        with get_db() as db:
            db.query(Provider).all()

    Or as a FastAPI dependency:
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db(drop_all: bool = False) -> None:
    """
    Initialize the database schema.

    Args:
        drop_all: If True, drop all tables before creating (for testing)
    """
    engine = get_engine()

    if drop_all:
        logger.warning("Dropping all database tables")
        Base.metadata.drop_all(bind=engine)

    logger.info("Creating database tables")
    Base.metadata.create_all(bind=engine)


def check_db_initialized() -> bool:
    """Check if the database has been initialized with tables."""
    engine = get_engine()
    from sqlalchemy import inspect

    inspector = inspect(engine)
    return "providers" in inspector.get_table_names()


def reset_connection() -> None:
    """Reset the database connection (useful for testing)."""
    global _engine, _SessionLocal

    if _engine:
        _engine.dispose()

    _engine = None
    _SessionLocal = None
