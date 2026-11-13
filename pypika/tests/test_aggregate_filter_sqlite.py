"""
Result-level cross-checks between native ``FILTER (WHERE ...)`` SQL and the
CASE-based fallback.

Both renderings are executed against SQLite (3.30.0+ understands the standard
FILTER clause), which proves the fallback is semantically equivalent rather
than merely syntactically plausible.
"""

import sqlite3
import unittest

from pypika import MySQLQuery, SQLLiteQuery, Table
from pypika import analytics as an
from pypika import functions as fn


class AggregateFilterSqliteEquivalenceTests(unittest.TestCase):
    table = Table("t")

    @classmethod
    def setUpClass(cls):
        if sqlite3.sqlite_version_info < (3, 30, 0):
            raise unittest.SkipTest("Native FILTER clause requires SQLite >= 3.30.0")

        cls.connection = sqlite3.connect(":memory:")
        cls.connection.execute("CREATE TABLE t (grp TEXT, id INTEGER, bar INTEGER)")
        # Deliberately includes:
        # - NULL filter inputs (id IS NULL) so predicates become UNKNOWN
        # - NULL aggregate inputs (bar IS NULL), including an all-NULL group
        # - duplicate bar values to make DISTINCT meaningful
        cls.connection.executemany(
            "INSERT INTO t VALUES (?, ?, ?)",
            [
                ("a", 1, 10),
                ("a", 2, 20),
                ("a", 3, None),
                ("a", None, 40),
                ("a", 5, 10),
                ("b", 1, 100),
                ("b", None, None),
                ("c", None, None),
                ("c", None, 7),
                ("c", 9, None),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        cls.connection.close()

    def _assert_equivalent(self, select_builder, expected_columns):
        native_query = select_builder(SQLLiteQuery)
        fallback_query = select_builder(MySQLQuery)

        self.assertIn("FILTER(WHERE", str(native_query))
        self.assertNotIn("FILTER(WHERE", str(fallback_query))

        native_rows = self.connection.execute(str(native_query)).fetchall()
        fallback_rows = self.connection.execute(str(fallback_query)).fetchall()

        self.assertEqual(expected_columns, len(native_rows[0]) if native_rows else 0)
        self.assertEqual(native_rows, fallback_rows)
        return native_rows

    def test_count_star(self):
        rows = self._assert_equivalent(
            lambda query: query.from_(self.table).select(fn.Count("*").filter(self.table.id > 1)),
            expected_columns=1,
        )
        self.assertEqual([(4,)], rows)

    def test_count_field_ignores_null_values(self):
        rows = self._assert_equivalent(
            lambda query: query.from_(self.table).select(fn.Count(self.table.bar).filter(self.table.id > 1)),
            expected_columns=1,
        )
        self.assertEqual([(2,)], rows)

    def test_sum_avg_min_max(self):
        aggregates = [
            fn.Sum(self.table.bar).filter(self.table.id > 1),
            fn.Avg(self.table.bar).filter(self.table.id > 1),
            fn.Min(self.table.bar).filter(self.table.id > 1),
            fn.Max(self.table.bar).filter(self.table.id > 1),
        ]
        self._assert_equivalent(
            lambda query: query.from_(self.table).select(*aggregates),
            expected_columns=4,
        )

    def test_distinct_aggregates(self):
        aggregates = [
            fn.Count(self.table.bar).distinct().filter(self.table.id > 1),
            fn.Sum(self.table.bar).distinct().filter(self.table.id > 1),
        ]
        rows = self._assert_equivalent(
            lambda query: query.from_(self.table).select(*aggregates),
            expected_columns=2,
        )
        # bar values surviving the filter: 20, NULL, 10 -> 2 distinct non-NULL,
        # distinct sum = 30.
        self.assertEqual([(2, 30)], rows)

    def test_null_filter_predicate_matches_nothing(self):
        # "id" = NULL is UNKNOWN on every row: COUNT(*) -> 0, SUM -> NULL.
        rows = self._assert_equivalent(
            lambda query: query.from_(self.table).select(
                fn.Count("*").filter(self.table.id == None),  # noqa: E711 - SQL three-valued logic
                fn.Sum(self.table.bar).filter(self.table.id == None),  # noqa: E711
            ),
            expected_columns=2,
        )
        self.assertEqual([(0, None)], rows)

    def test_all_null_inputs_group(self):
        # Group "c" has a row whose filter passes (id = 9) but whose aggregate
        # input (bar) is NULL: every filtered aggregate input is NULL. COUNT
        # ignores NULL inputs while the other aggregates must return NULL in
        # both renderings.
        rows = self._assert_equivalent(
            lambda query: query.from_(self.table)
            .where(self.table.grp == "c")
            .select(
                fn.Count("*").filter(self.table.id == 9),
                fn.Count(self.table.bar).filter(self.table.id == 9),
                fn.Sum(self.table.bar).filter(self.table.id == 9),
                fn.Avg(self.table.bar).filter(self.table.id == 9),
                fn.Min(self.table.bar).filter(self.table.id == 9),
                fn.Max(self.table.bar).filter(self.table.id == 9),
            ),
            expected_columns=6,
        )
        self.assertEqual([(1, 0, None, None, None, None)], rows)

    def test_empty_group_after_filter(self):
        # Grouping preserves both groups even though the filter never matches:
        # COUNT variants return 0, the rest NULL.
        rows = self._assert_equivalent(
            lambda query: query.from_(self.table)
            .groupby(self.table.grp)
            .select(
                self.table.grp,
                fn.Count("*").filter(self.table.id > 100),
                fn.Count(self.table.bar).filter(self.table.id > 100),
                fn.Sum(self.table.bar).filter(self.table.id > 100),
                fn.Avg(self.table.bar).filter(self.table.id > 100),
                fn.Min(self.table.bar).filter(self.table.id > 100),
                fn.Max(self.table.bar).filter(self.table.id > 100),
            )
            .orderby(self.table.grp),
            expected_columns=7,
        )
        self.assertEqual(
            [
                ("a", 0, 0, None, None, None, None),
                ("b", 0, 0, None, None, None, None),
                ("c", 0, 0, None, None, None, None),
            ],
            rows,
        )

    def test_grouped_aggregates_with_data(self):
        aggregates = [
            fn.Count("*").filter(self.table.id > 1),
            fn.Count(self.table.bar).filter(self.table.id > 1),
            fn.Sum(self.table.bar).filter(self.table.id > 1),
            fn.Avg(self.table.bar).filter(self.table.id > 1),
            fn.Min(self.table.bar).filter(self.table.id > 1),
            fn.Max(self.table.bar).filter(self.table.id > 1),
            fn.Count(self.table.bar).distinct().filter(self.table.id > 1),
        ]
        self._assert_equivalent(
            lambda query: query.from_(self.table)
            .groupby(self.table.grp)
            .select(self.table.grp, *aggregates)
            .orderby(self.table.grp),
            expected_columns=8,
        )

    def test_window_aggregate_with_filter(self):
        window = an.Sum(self.table.bar).filter(self.table.id > 1).over(self.table.grp)

        native_query = SQLLiteQuery.from_(self.table).select(self.table.grp, self.table.id, window).orderby(
            self.table.grp, self.table.id
        )
        fallback_query = MySQLQuery.from_(self.table).select(self.table.grp, self.table.id, window).orderby(
            self.table.grp, self.table.id
        )

        native_rows = self.connection.execute(str(native_query)).fetchall()
        fallback_rows = self.connection.execute(str(fallback_query)).fetchall()

        self.assertEqual(native_rows, fallback_rows)
        self.assertIn("FILTER(WHERE", str(native_query))
        self.assertNotIn("FILTER(WHERE", str(fallback_query))
