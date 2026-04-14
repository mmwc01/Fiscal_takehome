"""
Shared pytest fixtures.

tmp_db  — fresh SQLite database in a temp directory (per-test)
conn    — open connection to tmp_db, auto-committed after test
"""
import pytest

from fiscal import db


@pytest.fixture
def tmp_db(tmp_path):
    """Initialize a fresh database in a temporary directory."""
    path = tmp_path / "test.db"
    db.init_db(path)
    return path


@pytest.fixture
def conn(tmp_db):
    """
    Open connection to tmp_db. Commits when the test completes cleanly,
    rolls back on failure.
    """
    with db.get_connection(tmp_db) as c:
        yield c
