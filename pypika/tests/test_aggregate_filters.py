import sqlite3
import unittest

import pypika.analytics as an
from pypika import Query, Table, functions as fn
from pypika.dialects import (
    ClickHouseQuery,
    MSSQLQuery,
    MySQLQuery,
    OracleQuery,
    PostgreSQLQuery,
    RedshiftQuery,
    SQLLiteQuery,
    VerticaQuery,
)
from pypika.terms import AggregateFunction
from pypika.utils import FunctionException


class AggregateFilterSqlTests(unittest.TestCase):
    def setUp(self):
        self.table = Table("t")

    def test_default_and_postgres_and_sqlite_emit_native_filter(self):
        expr = fn.Sum(self.table.b).filter(self.table.a > 1)
        expected = 'SELECT SUM("b") FILTER(WHERE "a">1) FROM "t"'

        self.assertEqual(expected, str(Query.from_(self.table).select(expr)))
        self.assertEqual(expected, str(PostgreSQLQuery.from_(self.table).select(expr)))
        self.assertEqual(expected, str(SQLLiteQuery.from_(self.table).select(expr)))

    def test_dialects_without_native_support_emit_case_rewrite(self):
        expr = fn.Sum(self.table.b).filter(self.table.a > 1)

        self.assertEqual(
            "SELECT SUM(CASE WHEN `a`>1 THEN `b` END) FROM `t`",
            str(MySQLQuery.from_(self.table).select(expr)),
        )
        self.assertEqual(
            'SELECT SUM(CASE WHEN "a">1 THEN "b" END) FROM "t"',
            str(MSSQLQuery.from_(self.table).select(expr)),
        )
        self.assertEqual(
            'SELECT SUM(CASE WHEN "a">1 THEN "b" END) FROM "t"',
            str(VerticaQuery.from_(self.table).select(expr)),
        )
        self.assertEqual(
            'SELECT SUM(CASE WHEN "a">1 THEN "b" END) FROM "t"',
            str(RedshiftQuery.from_(self.table).select(expr)),
        )
        self.assertEqual(
            "SELECT SUM(CASE WHEN a>1 THEN b END) FROM t",
            str(OracleQuery.from_(self.table).select(expr)),
        )
        self.assertEqual(
            'SELECT SUM(CASE WHEN "a">1 THEN "b" END) FROM "t"',
            str(ClickHouseQuery.from_(self.table).select(expr)),
        )

    def test_count_star_fallback_uses_literal_one(self):
        expr = fn.Count("*").filter(self.table.a > 1)

        self.assertEqual('SELECT COUNT(CASE WHEN "a">1 THEN 1 END) FROM "t"', self._fallback_sql(expr))
        self.assertEqual(
            'SELECT COUNT(*) FILTER(WHERE "a">1) FROM "t"',
            str(SQLLiteQuery.from_(self.table).select(expr)),
        )

    def test_distinct_is_preserved_in_fallback(self):
        expr = fn.Count(self.table.b).distinct().filter(self.table.a > 1)

        self.assertEqual(
            'SELECT COUNT(DISTINCT CASE WHEN "a">1 THEN "b" END) FROM "t"',
            self._fallback_sql(expr),
        )
        self.assertEqual(
            'SELECT COUNT(DISTINCT "b") FILTER(WHERE "a">1) FROM "t"',
            str(SQLLiteQuery.from_(self.table).select(expr)),
        )

    def test_filter_with_null_criterion_keeps_three_valued_logic(self):
        expr = fn.Sum(self.table.b).filter(self.table.a.isnull())

        self.assertEqual(
            'SELECT SUM(CASE WHEN "a" IS NULL THEN "b" END) FROM "t"',
            self._fallback_sql(expr),
        )

    def test_custom_aggregate_without_equivalent_raises(self):
        expr = AggregateFunction("MEDIAN", self.table.b).filter(self.table.a > 1)

        with self.assertRaises(FunctionException):
            str(MySQLQuery.from_(self.table).select(expr))

        # Native dialects are not affected: no rewrite is needed when the syntax exists.
        self.assertEqual(
            'SELECT MEDIAN("b") FILTER(WHERE "a">1) FROM "t"',
            str(PostgreSQLQuery.from_(self.table).select(expr)),
        )

    def test_unsupported_analytic_raises_instead_of_case_rewrite(self):
        expr = an.LastValue(self.table.b).filter(self.table.a > 1).over(self.table.g)

        with self.assertRaises(FunctionException):
            str(MySQLQuery.from_(self.table).select(expr))

    def test_window_filter_native_and_fallback(self):
        expr = an.Sum(self.table.b).filter(self.table.a > 1).over(self.table.g)

        self.assertEqual(
            'SELECT SUM("b") FILTER(WHERE "a">1) OVER(PARTITION BY "g") FROM "t"',
            str(SQLLiteQuery.from_(self.table).select(expr)),
        )
        self.assertEqual(
            'SELECT SUM(CASE WHEN "a">1 THEN "b" END) OVER(PARTITION BY "g") FROM "t"',
            self._fallback_sql(expr),
        )

    def test_same_expression_renders_native_and_fallback_without_mutation(self):
        # The filter is stored on the expression; only the rendering dialect chooses the syntax.
        expr = fn.Sum(self.table.b).filter(self.table.a > 1)

        native = str(SQLLiteQuery.from_(self.table).select(expr))
        fallback = self._fallback_sql(expr)
        native_again = str(SQLLiteQuery.from_(self.table).select(expr))

        self.assertEqual('SELECT SUM("b") FILTER(WHERE "a">1) FROM "t"', native)
        self.assertEqual('SELECT SUM(CASE WHEN "a">1 THEN "b" END) FROM "t"', fallback)
        self.assertEqual(native, native_again)

    def _fallback_sql(self, expr):
        return str(Query.from_(self.table).select(expr).get_sql(supports_aggregate_filter=False))


class AggregateFilterSqliteEquivalenceTests(unittest.TestCase):
    """
    SQLite (>= 3.30) executes both the native FILTER(WHERE ...) clause and the CASE-based rewrite, so the two
    forms are compared directly against the same in-memory data.
    """

    ROWS = [
        # g, a, b — includes NULL inputs, NULL filter columns and groups left empty after filtering.
        ("x", 1, 10),
        ("x", 2, 20),
        ("x", 3, None),
        ("x", None, 40),
        ("y", 1, 50),
        ("y", None, 60),
        ("z", None, None),
    ]

    def setUp(self):
        self.table = Table("t")
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute('CREATE TABLE "t" ("g" VARCHAR, "a" INTEGER, "b" INTEGER)')
        self.connection.executemany('INSERT INTO "t" VALUES (?,?,?)', self.ROWS)

    def tearDown(self):
        self.connection.close()

    def test_native_filter_matches_case_rewrite(self):
        aggregates = [
            fn.Count("*"),
            fn.Count(self.table.b),
            fn.Count(self.table.b).distinct(),
            fn.Sum(self.table.b),
            fn.Avg(self.table.b),
            fn.Min(self.table.b),
            fn.Max(self.table.b),
        ]
        filters = [
            self.table.a > 1,
            self.table.a.isnull(),
            (self.table.a > 1) | self.table.a.isnull(),
            self.table.b > 15,
        ]

        for grouped in (False, True):
            for aggregate in aggregates:
                for criterion in filters:
                    with self.subTest(grouped=grouped, aggregate=aggregate.name, criterion=str(criterion)):
                        filtered = [aggregate.filter(criterion)]

                        native_sql = self._build(filtered, grouped, supports_aggregate_filter=True)
                        fallback_sql = self._build(filtered, grouped, supports_aggregate_filter=False)

                        self.assertEqual(
                            self.connection.execute(native_sql).fetchall(),
                            self.connection.execute(fallback_sql).fetchall(),
                        )

    def _build(self, selects, grouped, supports_aggregate_filter):
        query = SQLLiteQuery.from_(self.table)
        if grouped:
            query = query.groupby(self.table.g).select(self.table.g, *selects)
        else:
            query = query.select(*selects)

        return query.get_sql(supports_aggregate_filter=supports_aggregate_filter)


if __name__ == "__main__":
    unittest.main()
