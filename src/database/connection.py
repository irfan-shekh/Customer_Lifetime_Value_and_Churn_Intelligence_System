"""
Database connection factory and health verification module.
Manages PostgreSQL connections via SQLAlchemy/psycopg2 and provides fallback diagnostics.
"""

from typing import Dict, Any, Generator, Tuple, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
import logging

from src.config import DATABASE_URL
from src.database.schema import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_SessionFactory = None
_active_storage_mode: str = "UNKNOWN"
_connection_message: str = ""


def get_active_storage_mode() -> Tuple[str, str]:
    """Returns (storage_mode, diagnostic_message)."""
    global _active_storage_mode, _connection_message
    if _active_storage_mode == "UNKNOWN":
        _init_engine()
    return _active_storage_mode, _connection_message


def _mask_url(url: str) -> str:
    """Masks database password in connection string for safe display."""
    if "@" in url and "://" in url:
        prefix, rest = url.split("://", 1)
        creds, host_part = rest.split("@", 1)
        if ":" in creds:
            user, _ = creds.split(":", 1)
            return f"{prefix}://{user}:*****@{host_part}"
    return url


def _init_engine():
    global _engine, _SessionFactory, _active_storage_mode, _connection_message

    # Dedicated PostgreSQL connection
    if DATABASE_URL and DATABASE_URL.startswith("postgres"):
        candidate_urls = [DATABASE_URL]
        # Support both psycopg (v3) and psycopg2 (v2) drivers
        if DATABASE_URL.startswith("postgresql://"):
            candidate_urls.append(DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1))
            candidate_urls.append(DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1))

        last_error = None
        for url in candidate_urls:
            try:
                logger.info(f"Attempting connection to PostgreSQL via {_mask_url(url)}...")
                pg_engine = create_engine(
                    url,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    connect_args={"connect_timeout": 5},
                )
                # Test connection health
                with pg_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                _engine = pg_engine
                _SessionFactory = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
                _active_storage_mode = "POSTGRESQL"
                _connection_message = f"Connected to PostgreSQL ({_mask_url(url)})"
                logger.info(_connection_message)
                return
            except Exception as e:
                last_error = e
                continue

        logger.error(f"PostgreSQL connection to DATABASE_URL failed: {last_error}")
        _engine = None
        _SessionFactory = None
        _active_storage_mode = "DISCONNECTED"
        _connection_message = (
            f"PostgreSQL unreachable ({type(last_error).__name__}): {last_error}. "
            "Ensure PostgreSQL 18 is running and verify DATABASE_URL in .env."
        )
    else:
        _engine = None
        _SessionFactory = None
        _active_storage_mode = "DISCONNECTED"
        _connection_message = (
            "DATABASE_URL is not set or empty. "
            "Please configure a valid PostgreSQL connection string in .env."
        )
        logger.warning(_connection_message)


def get_engine() -> Optional[Engine]:
    """Returns the active SQLAlchemy Engine or None if uninitialized."""
    if _engine is None:
        _init_engine()
    return _engine


def get_db_session() -> Generator[Session, None, None]:
    """Yields a managed database session."""
    if _SessionFactory is None:
        _init_engine()
    if _SessionFactory is None:
        raise ConnectionError(
            "PostgreSQL session factory is uninitialized. "
            f"Reason: {_connection_message}"
        )
    session = _SessionFactory()
    try:
        yield session
    finally:
        session.close()


def initialize_schema():
    """Initializes tables for active database engine."""
    engine = get_engine()
    if engine is None:
        logger.warning("PostgreSQL schema initialization skipped: database engine is uninitialized.")
        return
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully.")


def check_connection_health() -> Dict[str, Any]:
    """Runs a live health check on the active database."""
    mode, msg = get_active_storage_mode()
    if _engine is None:
        return {
            "status": "DISCONNECTED",
            "storage_mode": mode,
            "database_url": _mask_url(DATABASE_URL) if DATABASE_URL else "Not Configured",
            "message": msg,
            "tables_registered": list(Base.metadata.tables.keys()),
        }
    try:
        with _engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            is_healthy = (result == 1)
        return {
            "status": "HEALTHY" if is_healthy else "UNHEALTHY",
            "storage_mode": mode,
            "database_url": _mask_url(str(_engine.url)),
            "message": msg,
            "tables_registered": list(Base.metadata.tables.keys()),
        }
    except Exception as e:
        return {
            "status": "UNHEALTHY",
            "storage_mode": mode,
            "database_url": _mask_url(str(_engine.url)),
            "message": f"Health check failed: {str(e)}",
            "tables_registered": list(Base.metadata.tables.keys()),
        }
