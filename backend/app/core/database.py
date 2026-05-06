from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from contextlib import contextmanager
from .config import settings

engine = create_engine(
    settings.sqlite_url,
    connect_args={"check_same_thread": False},
    echo=False,
)

# 开启 WAL 模式，提升并发读写性能
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_con, _):
    cursor = dbapi_con.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Session:
    """FastAPI Depends 注入用"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session():
    """非 FastAPI 上下文（脚本/启动事件）使用"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_all_tables():
    from app.models import gpu_spec, model, benchmark_run, benchmark_data  # noqa: F401
    Base.metadata.create_all(bind=engine)
