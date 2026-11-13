"""
Dialect capability registry.

Centralizes which optional SQL features each target dialect supports so that
SQL generation can pick native syntax or a semantically equivalent fallback.
Currently the only declared capability is the standard ``AGGREGATE_FILTER``
clause (SQL:2003, ``agg(...) FILTER(WHERE ...)``).
"""

from __future__ import annotations

from pypika.enums import Dialects

__author__ = "Timothy Heys"
__email__ = "theys@kayak.com"


class DialectCapabilities:
    """
    Describes the optional SQL features supported by a dialect.

    Capabilities default to ``False`` so that adding a new feature requires an
    explicit opt-in per dialect instead of silently assuming every database
    supports it.
    """

    def __init__(self, aggregate_filter: bool = False) -> None:
        self.aggregate_filter = aggregate_filter


# Dialects whose engines implement the standard FILTER (WHERE ...) clause.
# - PostgreSQL 9.4+, SQLite 3.30.0+ and Snowflake support it natively.
# - MySQL (all versions), Oracle, SQL Server, Redshift, Vertica, ClickHouse
#   and Jira do not, so callers must use the CASE-based fallback.
_CAPABILITIES = {
    Dialects.POSTGRESQL: DialectCapabilities(aggregate_filter=True),
    Dialects.SQLLITE: DialectCapabilities(aggregate_filter=True),
    Dialects.SNOWFLAKE: DialectCapabilities(aggregate_filter=True),
}

# Preserves the historical default: a term rendered without a concrete dialect
# (e.g. ``str(fn.Sum("x").filter(...))``) emits the standard FILTER clause.
_DEFAULT_CAPABILITIES = DialectCapabilities(aggregate_filter=True)


def get_capabilities(dialect: Dialects | None) -> DialectCapabilities:
    """
    Returns the capability set for ``dialect``.

    Unknown or ``None`` dialects get the standard-SQL default capabilities.
    """
    if dialect is None:
        return _DEFAULT_CAPABILITIES
    return _CAPABILITIES.get(dialect, DialectCapabilities())


def supports_aggregate_filter(dialect: Dialects | None) -> bool:
    return get_capabilities(dialect).aggregate_filter
