from __future__ import annotations

from typing import Any, Iterator, Sequence

__author__ = "Timothy Heys"
__email__ = "theys@kayak.com"


UNKNOWN = object()
"""
Sentinel marking the part of an output column contract that cannot be determined statically.

Projections which cannot be expanded without catalog information, such as ``*`` or ``table.*``, keep
``UNKNOWN`` instead of guessing column names or counts.
"""


class OutputColumn:
    """
    A statically derived description of one output column of a query.

    ``name`` is the output name used by outer references (ORDER BY of a set operation, a wrapping
    subquery, ...). It is the explicit alias when one is given, otherwise the field name for plain
    field projections, otherwise the rendered expression text. ``name`` is ``None`` when the column
    cannot be named statically (currently only inside an unknown expansion such as ``*``).
    """

    __slots__ = ("name", "expression")

    def __init__(self, name: str | None, expression: Any = None) -> None:
        self.name = name
        self.expression = expression

    @property
    def known(self) -> bool:
        return self.name is not None

    def __repr__(self) -> str:
        return "OutputColumn(name={!r})".format(self.name)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, OutputColumn):
            return self.name == other.name
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)


class OutputColumnContract:
    """
    An inspectable description of the output columns of a query.

    The contract is a sequence of :class:`OutputColumn` entries in projection order. Duplicate output
    names are preserved; every position keeps its own entry. When any part of the projection cannot be
    expanded statically (``*``, ``table.*``), the whole contract is marked unknown: no column count or
    names are guessed.
    """

    __slots__ = ("columns", "unknown_reason")

    def __init__(self, columns: Sequence[OutputColumn], unknown_reason: str | None = None) -> None:
        self.columns = tuple(columns)
        self.unknown_reason = unknown_reason

    @classmethod
    def unknown(cls, reason: str) -> OutputColumnContract:
        return cls((), unknown_reason=reason)

    @property
    def known(self) -> bool:
        return self.unknown_reason is None

    @property
    def names(self) -> tuple[str, ...] | object:
        """
        The output names in projection order, or ``UNKNOWN`` when the contract cannot be expanded
        statically.
        """
        if not self.known:
            return UNKNOWN
        return tuple(column.name for column in self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def __iter__(self) -> Iterator[OutputColumn]:
        return iter(self.columns)

    def __getitem__(self, index: int) -> OutputColumn:
        return self.columns[index]

    def __repr__(self) -> str:
        if not self.known:
            return "OutputColumnContract(UNKNOWN: {})".format(self.unknown_reason)
        return "OutputColumnContract(names={!r})".format(self.names)

    def duplicates(self) -> tuple[str, ...]:
        """Output names which occur more than once, in first-seen order."""
        seen: set[str] = set()
        duplicated: list[str] = []
        for column in self.columns:
            if column.name in seen and column.name not in duplicated:
                duplicated.append(column.name)
            seen.add(column.name)
        return tuple(duplicated)

    def find_name(self, name: str) -> list[int]:
        """All 0-based positions whose output name matches ``name``."""
        return [index for index, column in enumerate(self.columns) if column.name == name]


def from_terms(terms: Sequence[Any], quote_char: str | None = None) -> OutputColumnContract:
    """
    Builds an output column contract from a query's projected terms, without executing any SQL.

    Any term which cannot be expanded statically (``*`` / ``table.*``) makes the whole contract
    unknown instead of guessing the expanded columns.
    """
    from pypika.terms import Star

    columns: list[OutputColumn] = []
    for term in terms:
        if isinstance(term, Star):
            return OutputColumnContract.unknown(
                "projection {} cannot be expanded statically".format(term.get_sql(quote_char=quote_char))
            )

        name = term.output_name(quote_char=quote_char)
        if name is None:
            return OutputColumnContract.unknown("projection cannot be expanded statically")

        columns.append(OutputColumn(name=name, expression=term))

    if not columns:
        return OutputColumnContract.unknown("query has no projected columns")

    return OutputColumnContract(columns)
