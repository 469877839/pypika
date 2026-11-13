import unittest

from pypika import (
    ClickHouseQuery,
    MSSQLQuery,
    MySQLQuery,
    OracleQuery,
    PostgreSQLQuery,
    Query,
    RedshiftQuery,
    SQLLiteQuery,
    Table,
    VerticaQuery,
)
from pypika import analytics as an
from pypika import functions as fn
from pypika.dialect_capabilities import get_capabilities, supports_aggregate_filter
from pypika.enums import Dialects
from pypika.terms import AggregateFunction
from pypika.utils import FunctionException


class AggregateFilterDialectCapabilityTests(unittest.TestCase):
    def test_native_filter_dialects(self):
        for dialect in (Dialects.POSTGRESQL, Dialects.SQLLITE, Dialects.SNOWFLAKE):
            with self.subTest(dialect=dialect):
                self.assertTrue(supports_aggregate_filter(dialect))
                self.assertTrue(get_capabilities(dialect).aggregate_filter)

    def test_fallback_dialects(self):
        for dialect in (
            Dialects.MYSQL,
            Dialects.ORACLE,
            Dialects.MSSQL,
            Dialects.REDSHIFT,
            Dialects.VERTICA,
            Dialects.CLICKHOUSE,
            Dialects.JIRA,
        ):
            with self.subTest(dialect=dialect):
                self.assertFalse(supports_aggregate_filter(dialect))

    def test_unknown_dialect_defaults_to_standard_sql(self):
        # Rendering a term without a concrete dialect historically emits the
        # standard FILTER clause, so None must keep that behavior.
        self.assertTrue(supports_aggregate_filter(None))


class AggregateFilterSqlTests(unittest.TestCase):
    table = Table("abc")

    def sql(self, expr, dialect):
        return expr.get_sql(dialect=dialect, quote_char='"')

    def test_native_filter_emitted_for_supporting_dialect(self):
        expr = fn.Sum(self.table.bar).filter(self.table.id.eq(1))

        self.assertEqual('SUM("bar") FILTER(WHERE "id"=1)', self.sql(expr, Dialects.SQLLITE))

    def test_native_filter_default_dialect(self):
        expr = fn.Sum(self.table.bar).filter(self.table.id.eq(1))

        self.assertEqual('SUM("bar") FILTER(WHERE "id"=1)', str(expr))

    def test_fallback_count_star(self):
        expr = fn.Count("*").filter(self.table.id.eq(1))

        self.assertEqual(
            "SUM(CASE WHEN \"id\"=1 THEN 1 ELSE 0 END)",
            self.sql(expr, Dialects.MYSQL),
        )

    def test_fallback_count_field(self):
        expr = fn.Count(self.table.bar).filter(self.table.id.eq(1))

        self.assertEqual(
            'COUNT(CASE WHEN "id"=1 THEN "bar" END)',
            self.sql(expr, Dialects.MYSQL),
        )

    def test_fallback_count_distinct(self):
        expr = fn.Count(self.table.bar).distinct().filter(self.table.id.eq(1))

        self.assertEqual(
            'COUNT(DISTINCT CASE WHEN "id"=1 THEN "bar" END)',
            self.sql(expr, Dialects.MYSQL),
        )

    def test_fallback_sum_distinct(self):
        expr = fn.Sum(self.table.bar).distinct().filter(self.table.id.eq(1))

        self.assertEqual(
            'SUM(DISTINCT CASE WHEN "id"=1 THEN "bar" END)',
            self.sql(expr, Dialects.MYSQL),
        )

    def test_fallback_avg_min_max(self):
        expected = {
            "AVG": 'AVG(CASE WHEN "id"=1 THEN "bar" END)',
            "MIN": 'MIN(CASE WHEN "id"=1 THEN "bar" END)',
            "MAX": 'MAX(CASE WHEN "id"=1 THEN "bar" END)',
        }
        for aggregate, expected_sql in expected.items():
            with self.subTest(aggregate=aggregate):
                expr = getattr(fn, aggregate.capitalize())(self.table.bar).filter(self.table.id.eq(1))
                self.assertEqual(expected_sql, self.sql(expr, Dialects.MYSQL))

    def test_fallback_filter_with_null_comparison_uses_three_valued_logic(self):
        # "id" = NULL is UNKNOWN for every row, so the WHEN test simply never
        # matches just like the native FILTER clause.
        expr = fn.Sum(self.table.bar).filter(self.table.id.eq(None))

        self.assertEqual(
            'SUM(CASE WHEN "id"=null THEN "bar" END)',
            self.sql(expr, Dialects.MYSQL),
        )

    def test_fallback_combined_with_window(self):
        expr = an.Sum(self.table.bar).filter(self.table.id.eq(1)).over(self.table.foo)

        self.assertEqual(
            'SUM(CASE WHEN "id"=1 THEN "bar" END) OVER(PARTITION BY "foo")',
            self.sql(expr, Dialects.MYSQL),
        )

    def test_native_combined_with_window(self):
        expr = an.Sum(self.table.bar).filter(self.table.id.eq(1)).over(self.table.foo)

        self.assertEqual(
            'SUM("bar") FILTER(WHERE "id"=1) OVER(PARTITION BY "foo")',
            self.sql(expr, Dialects.SQLLITE),
        )

    def test_unsupported_custom_aggregate_raises(self):
        expr = AggregateFunction("STDDEV", self.table.bar).filter(self.table.id.eq(1))

        with self.assertRaises(FunctionException):
            self.sql(expr, Dialects.MYSQL)

    def test_unsupported_custom_aggregate_still_native_when_supported(self):
        # On a dialect with FILTER, arbitrary aggregates need no rewrite.
        expr = AggregateFunction("STDDEV", self.table.bar).filter(self.table.id.eq(1))

        self.assertEqual(
            'STDDEV("bar") FILTER(WHERE "id"=1)',
            self.sql(expr, Dialects.SQLLITE),
        )

    def test_unfiltered_aggregate_unaffected(self):
        expr = fn.StdDev(self.table.bar)

        self.assertEqual('STDDEV("bar")', self.sql(expr, Dialects.MYSQL))

    def test_dialect_query_builders_select_native_or_fallback(self):
        filtered = fn.Sum(self.table.bar).filter(self.table.id.eq(1))

        native = SQLLiteQuery.from_(self.table).select(filtered)
        fallback = MySQLQuery.from_(self.table).select(filtered)

        self.assertIn("FILTER(WHERE", str(native))
        self.assertIn("CASE WHEN", str(fallback))
        self.assertNotIn("FILTER(WHERE", str(fallback))

    def test_all_dialect_query_builders(self):
        filtered = fn.Count("*").filter(self.table.id.eq(1))

        native_builders = (
            PostgreSQLQuery.from_(self.table).select(filtered),
            SQLLiteQuery.from_(self.table).select(filtered),
        )
        fallback_builders = (
            MySQLQuery.from_(self.table).select(filtered),
            MSSQLQuery.from_(self.table).select(filtered),
            OracleQuery.from_(self.table).select(filtered),
            RedshiftQuery.from_(self.table).select(filtered),
            VerticaQuery.from_(self.table).select(filtered),
            ClickHouseQuery.from_(self.table).select(filtered),
        )

        for query in native_builders:
            with self.subTest(query=query):
                self.assertIn("FILTER(WHERE", str(query))
        for query in fallback_builders:
            with self.subTest(query=query):
                self.assertNotIn("FILTER(WHERE", str(query))
                self.assertIn("CASE WHEN", str(query))


class AggregateFilterReuseTests(unittest.TestCase):
    table = Table("abc")

    def test_same_term_reused_across_dialects(self):
        # A single metric definition is rendered for both a FILTER-capable and
        # a FILTER-incapable database without mutating the shared term.
        filtered = fn.Sum(self.table.bar).filter(self.table.id.eq(1))

        native_query = SQLLiteQuery.from_(self.table).select(filtered)
        fallback_query = MySQLQuery.from_(self.table).select(filtered)

        self.assertIn("FILTER(WHERE", str(native_query))
        self.assertIn("CASE WHEN", str(fallback_query))
        # Re-rendering the native query still works, proving no shared state
        # was mutated while producing the fallback SQL.
        self.assertIn("FILTER(WHERE", str(native_query))

    def test_filter_is_immutable_builder(self):
        original = fn.Sum(self.table.bar)
        filtered = original.filter(self.table.id.eq(1))

        self.assertIsNot(original, filtered)
        self.assertEqual('SUM("bar")', str(original))
        self.assertIn("FILTER(WHERE", str(filtered))

    def test_query_object_reused_between_dialects(self):
        query = Query.from_(self.table).select(
            fn.Count("*").filter(self.table.id.gt(0)),
            fn.Avg(self.table.bar).filter(self.table.id.gt(0)),
        )

        native_sql = query.get_sql(dialect=Dialects.SQLLITE)
        fallback_sql = query.get_sql(dialect=Dialects.MYSQL)

        self.assertIn("FILTER(WHERE", native_sql)
        self.assertNotIn("FILTER(WHERE", fallback_sql)
        # The original query object keeps the standard-SQL rendering.
        self.assertEqual(native_sql, query.get_sql(dialect=Dialects.SQLLITE))
