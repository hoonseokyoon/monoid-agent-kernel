from __future__ import annotations

import os

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service]


def _selected() -> bool:
    return os.environ.get("MONOID_SERVICE_PROFILE") in {"postgres", "objectstore", "combined"}


if not _selected():
    pytest.skip("PostgreSQL service profile is not selected", allow_module_level=True)

import psycopg  # noqa: E402


@pytest.mark.parametrize(
    ("dsn_variable", "expected_major", "profiles"),
    [
        ("MONOID_POSTGRES16_DSN", 16, {"postgres", "objectstore", "combined"}),
        ("MONOID_POSTGRES18_DSN", 18, {"combined"}),
    ],
)
def test_pinned_postgresql_service_is_reachable(
    dsn_variable: str,
    expected_major: int,
    profiles: set[str],
) -> None:
    if os.environ["MONOID_SERVICE_PROFILE"] not in profiles:
        pytest.skip(f"PostgreSQL {expected_major} is outside the selected profile")
    dsn = os.environ.get(dsn_variable)
    if not dsn:
        pytest.fail(f"{dsn_variable} is required for the selected service profile")

    with psycopg.connect(dsn, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_setting('server_version_num')")
            version_number = int(cursor.fetchone()[0])
            cursor.execute("SELECT 1")
            assert cursor.fetchone() == (1,)

    assert version_number // 10000 == expected_major
