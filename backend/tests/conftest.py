"""
pytest conftest.py — shared fixtures for all tests.
Uses an in-memory SQLite database and an ASGI test client.
"""
import asyncio

import fastapi.dependencies.utils
import fastapi.routing
import httpx
import pytest
import starlette.concurrency
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


async def _run_sync_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


# The local Starlette/anyio test stack hangs when sync endpoints or dependencies
# enter the threadpool. These in-process tests run sync callables inline instead.
fastapi.routing.run_in_threadpool = _run_sync_inline
fastapi.dependencies.utils.run_in_threadpool = _run_sync_inline
starlette.concurrency.run_in_threadpool = _run_sync_inline

from app.main import app
from app.core.cache import clear_prediction_cache
from app.core.database import get_db, Base
from app.seed_data import seed_initial_data


# In-memory SQLite for tests
TEST_DB_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class ASGISyncClient:
    """Small sync wrapper around httpx ASGITransport for local API tests."""

    def __init__(self, app):
        self._app = app

    def request(self, method: str, url: str, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(send())

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs):
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs):
        return self.request("DELETE", url, **kwargs)

    def close(self):
        pass


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Create all tables once per test session."""
    from app import models  # noqa: F401 - register SQLAlchemy models before create_all

    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session(setup_database):
    """Provide a transactional DB session that rolls back after each test."""
    clear_prediction_cache()
    connection = engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        clear_prediction_cache()


@pytest.fixture()
def client(db_session):
    """ASGI test client with the in-memory DB injected via dependency override."""
    async def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # Seed GPU + model metadata into the test DB
    seed_initial_data(db_session)
    db_session.commit()

    c = ASGISyncClient(app)
    try:
        yield c
    finally:
        c.close()

    app.dependency_overrides.clear()
