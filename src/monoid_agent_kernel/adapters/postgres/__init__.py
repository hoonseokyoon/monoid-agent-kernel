"""PostgreSQL adapter namespace.

The concrete adapter is introduced incrementally during v0.23.  Importing this namespace never
loads psycopg; implementation modules own that optional dependency boundary.
"""

__all__: list[str] = []
