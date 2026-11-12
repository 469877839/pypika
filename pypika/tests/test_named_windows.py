import sqlite3
import unittest
from copy import copy, deepcopy

from pypika import Order, Query, Table, Window, WindowException
from pypika import analytics as an
from pypika.dialects import MSSQLQuery, MySQLQuery, OracleQuery, PostgreSQLQuery, SQLLiteQuery


class NamedWindowSQLTests(unittest.TestCase):
    table = Table("events")

    def test_multiple_functions_share_one_named_window(self):
        query = (
            Query.from_(self.table)
            .window("w", partition=[self.table.g], orderby=[self.table.d])
            .select(
                an.Rank().over(window="w").as_("r"),
                an.Sum(self.table.v).over(window="w").as_("running"),
            )
        )

        self.assertEqual(
            'SELECT RANK() OVER "w" "r",SUM("v") OVER "w" "running" '
            'FROM "events" WINDOW "w" AS (PARTITION BY "g" ORDER BY "d")',
            str(query),
        )

    def test_window_definition_with_multiple_partition_and_orderby_terms(self):
        query = (
            Query.from_(self.table)
            .window("w", partition=[self.table.g, self.table.h], orderby=[(self.table.d, Order.desc), self.table.v])
            .select(an.Rank().over(window="w").as_("r"))
        )

        self.assertEqual(
            'SELECT RANK() OVER "w" "r" FROM "events" '
            'WINDOW "w" AS (PARTITION BY "g","h" ORDER BY "d" DESC,"v")',
            str(query),
        )

    def test_window_with_frame_definition(self):
        query = (
            Query.from_(self.table)
            .window(
                "w",
                partition=[self.table.g],
                orderby=[self.table.d],
                frame="ROWS",
                bound=(an.Preceding(), an.CURRENT_ROW),
            )
            .select(an.Sum(self.table.v).over(window="w").as_("s"))
        )

        self.assertEqual(
            'SELECT SUM("v") OVER "w" "s" FROM "events" '
            'WINDOW "w" AS (PARTITION BY "g" ORDER BY "d" ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)',
            str(query),
        )

    def test_two_level_window_inheritance(self):
        query = (
            Query.from_(self.table)
            .window("w1", partition=[self.table.g])
            .window("w2", base="w1", orderby=[self.table.d])
            .window("w3", base="w2", frame="ROWS", bound=an.Preceding())
            .select(
                an.Rank().over(window="w2").as_("r"),
                an.Sum(self.table.v).over(window="w3").as_("s"),
            )
        )

        self.assertEqual(
            'SELECT RANK() OVER "w2" "r",SUM("v") OVER "w3" "s" FROM "events" '
            'WINDOW "w1" AS (PARTITION BY "g"),'
            '"w2" AS ("w1" ORDER BY "d"),'
            '"w3" AS ("w2" ROWS UNBOUNDED PRECEDING)',
            str(query),
        )

    def test_window_definitions_emitted_before_their_bases_are_reordered(self):
        query = (
            Query.from_(self.table)
            .window("w2", base="w1", orderby=[self.table.d])
            .window("w1", partition=[self.table.g])
            .select(an.Rank().over(window="w2").as_("r"))
        )

        self.assertEqual(
            'SELECT RANK() OVER "w2" "r" FROM "events" '
            'WINDOW "w1" AS (PARTITION BY "g"),"w2" AS ("w1" ORDER BY "d")',
            str(query),
        )

    def test_analytic_function_refines_named_window_with_orderby(self):
        query = (
            Query.from_(self.table)
            .window("w", partition=[self.table.g])
            .select(an.Rank().over(window="w").orderby(self.table.d, order=Order.desc).as_("r"))
        )

        self.assertEqual(
            'SELECT RANK() OVER("w" ORDER BY "d" DESC) "r" FROM "events" '
            'WINDOW "w" AS (PARTITION BY "g")',
            str(query),
        )

    def test_analytic_function_refines_named_window_with_frame(self):
        query = (
            Query.from_(self.table)
            .window("w", partition=[self.table.g], orderby=[self.table.d])
            .select(an.Sum(self.table.v).over(window="w").rows(an.Preceding()).as_("s"))
        )

        self.assertEqual(
            'SELECT SUM("v") OVER("w" ROWS UNBOUNDED PRECEDING) "s" FROM "events" '
            'WINDOW "w" AS (PARTITION BY "g" ORDER BY "d")',
            str(query),
        )

    def test_named_window_used_in_query_orderby(self):
        query = (
            Query.from_(self.table)
            .window("w", partition=[self.table.g], orderby=[self.table.d])
            .select(self.table.g, self.table.d)
            .orderby(an.Rank().over(window="w"))
        )

        self.assertEqual(
            'SELECT "g","d" FROM "events" '
            'WINDOW "w" AS (PARTITION BY "g" ORDER BY "d") ORDER BY RANK() OVER "w"',
            str(query),
        )

    def test_subquery_shadows_outer_window_name(self):
        sub = Table("sub")
        inner = (
            Query.from_(self.table)
            .window("w", partition=[self.table.g], orderby=[self.table.d])
            .select(an.Rank().over(window="w").as_("r"), self.table.g, self.table.d)
        )
        query = (
            Query.from_(inner.as_("sub"))
            .window("w", partition=[sub.g], orderby=[sub.r])
            .select(sub.g, an.DenseRank().over(window="w").as_("dr"))
        )

        self.assertEqual(
            'SELECT "sub"."g",DENSE_RANK() OVER "w" "dr" '
            'FROM (SELECT RANK() OVER "w" "r","g","d" FROM "events" '
            'WINDOW "w" AS (PARTITION BY "g" ORDER BY "d")) "sub" '
            'WINDOW "w" AS (PARTITION BY "sub"."g" ORDER BY "sub"."r")',
            str(query),
        )

    def test_window_name_does_not_leak_across_subquery_scope(self):
        inner = (
            Query.from_(self.table)
            .select(an.Rank().over(window="outer_w").as_("r"))
        )
        query = Query.from_(inner).window("outer_w", partition=[self.table.g]).select("r")

        with self.assertRaises(WindowException) as ctx:
            str(query)
        self.assertIn("outer_w", str(ctx.exception))


class NamedWindowDialectTests(unittest.TestCase):
    table = Table("events")

    def _shared_query(self, query_cls):
        return (
            query_cls.from_(self.table)
            .window("w", partition=[self.table.g], orderby=[self.table.d])
            .select(
                an.Rank().over(window="w").as_("r"),
                an.Sum(self.table.v).over(window="w").as_("running"),
            )
        )

    def test_postgresql_emits_native_window_clause(self):
        self.assertEqual(
            'SELECT RANK() OVER "w" "r",SUM("v") OVER "w" "running" '
            'FROM "events" WINDOW "w" AS (PARTITION BY "g" ORDER BY "d")',
            str(self._shared_query(PostgreSQLQuery)),
        )

    def test_mysql_emits_native_window_clause_with_backticks(self):
        self.assertEqual(
            "SELECT RANK() OVER `w` `r`,SUM(`v`) OVER `w` `running` "
            "FROM `events` WINDOW `w` AS (PARTITION BY `g` ORDER BY `d`)",
            str(self._shared_query(MySQLQuery)),
        )

    def test_mssql_expands_named_windows_without_unbound_references(self):
        query = (
            MSSQLQuery.from_(self.table)
            .window("w1", partition=[self.table.g])
            .window("w2", base="w1", orderby=[self.table.d])
            .window("w3", base="w2", frame="ROWS", bound=an.Preceding())
            .select(
                an.Rank().over(window="w2").as_("r"),
                an.Sum(self.table.v).over(window="w3").as_("s"),
            )
        )

        sql = str(query)
        self.assertNotIn("WINDOW", sql)
        self.assertNotIn("OVER \"w", sql)
        self.assertEqual(
            'SELECT RANK() OVER(PARTITION BY "g" ORDER BY "d") "r",'
            'SUM("v") OVER(PARTITION BY "g" ORDER BY "d" ROWS UNBOUNDED PRECEDING) "s" '
            'FROM "events"',
            sql,
        )

    def test_mssql_expands_function_level_orderby_refinement(self):
        query = (
            MSSQLQuery.from_(self.table)
            .window("w", partition=[self.table.g])
            .select(an.Rank().over(window="w").orderby(self.table.d, order=Order.desc).as_("r"))
        )

        self.assertEqual(
            'SELECT RANK() OVER(PARTITION BY "g" ORDER BY "d" DESC) "r" FROM "events"',
            str(query),
        )

    def test_mssql_expands_frame_defined_on_named_window(self):
        query = (
            MSSQLQuery.from_(self.table)
            .window(
                "w",
                partition=[self.table.g],
                orderby=[self.table.d],
                frame="ROWS",
                bound=(an.Preceding(), an.CURRENT_ROW),
            )
            .select(an.Sum(self.table.v).over(window="w").as_("s"))
        )

        self.assertEqual(
            'SELECT SUM("v") OVER(PARTITION BY "g" ORDER BY "d" '
            'ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "s" FROM "events"',
            str(query),
        )

    def test_oracle_expands_named_windows_without_unbound_references(self):
        query = (
            OracleQuery.from_(self.table)
            .window("w1", partition=[self.table.g])
            .window("w2", base="w1", orderby=[self.table.d])
            .select(an.Rank().over(window="w2").as_("r"))
        )

        sql = str(query)
        self.assertNotIn("WINDOW", sql)
        self.assertEqual(
            "SELECT RANK() OVER(PARTITION BY g ORDER BY d) r FROM events",
            sql,
        )


class NamedWindowValidationTests(unittest.TestCase):
    table = Table("events")

    def test_duplicate_window_name_rejected_at_definition(self):
        builder = Query.from_(self.table).window("w", partition=[self.table.g])

        with self.assertRaises(WindowException) as ctx:
            builder.window("w", partition=[self.table.d])

        self.assertIn("'w'", str(ctx.exception))
        self.assertIn("already defined", str(ctx.exception))

    def test_unknown_base_window_rejected_before_sql_is_generated(self):
        query = (
            Query.from_(self.table)
            .window("w2", base="w1", orderby=[self.table.d])
            .select(an.Rank().over(window="w2").as_("r"))
        )

        with self.assertRaises(WindowException) as ctx:
            str(query)
        message = str(ctx.exception)
        self.assertIn("'w2'", message)
        self.assertIn("'w1'", message)
        self.assertIn("unknown base window", message)

    def test_unknown_window_reference_rejected_before_sql_is_generated(self):
        query = Query.from_(self.table).select(an.Rank().over(window="nope").as_("r"))

        with self.assertRaises(WindowException) as ctx:
            str(query)
        self.assertIn("'nope'", str(ctx.exception))

    def test_cyclic_window_inheritance_reports_the_chain(self):
        query = (
            Query.from_(self.table)
            .window("w1", base="w3", partition=[self.table.g])
            .window("w2", base="w1", orderby=[self.table.d])
            .window("w3", base="w2", frame="ROWS", bound=an.Preceding())
            .select(an.Rank().over(window="w1").as_("r"))
        )

        with self.assertRaises(WindowException) as ctx:
            str(query)
        message = str(ctx.exception)
        self.assertIn("Cyclic window inheritance", message)
        self.assertIn("w1", message)
        self.assertIn("w2", message)
        self.assertIn("w3", message)

    def test_self_referencing_window_is_a_cycle(self):
        query = (
            Query.from_(self.table)
            .window("w", base="w")
            .select(an.Rank().over(window="w").as_("r"))
        )

        with self.assertRaises(WindowException) as ctx:
            str(query)
        self.assertIn("w -> w", str(ctx.exception))

    def test_child_window_cannot_override_parent_partition_by(self):
        query = (
            Query.from_(self.table)
            .window("w1", partition=[self.table.g])
            .window("w2", base="w1", partition=[self.table.d])
            .select(an.Rank().over(window="w2").as_("r"))
        )

        with self.assertRaises(WindowException) as ctx:
            str(query)
        message = str(ctx.exception)
        self.assertIn("'w2'", message)
        self.assertIn("PARTITION BY", message)

    def test_child_window_cannot_override_parent_order_by(self):
        query = (
            Query.from_(self.table)
            .window("w1", partition=[self.table.g], orderby=[self.table.d])
            .window("w2", base="w1", orderby=[self.table.v])
            .select(an.Rank().over(window="w2").as_("r"))
        )

        with self.assertRaises(WindowException) as ctx:
            str(query)
        message = str(ctx.exception)
        self.assertIn("'w2'", message)
        self.assertIn("ORDER BY", message)

    def test_child_window_cannot_override_parent_frame(self):
        query = (
            Query.from_(self.table)
            .window(
                "w1",
                partition=[self.table.g],
                orderby=[self.table.d],
                frame="ROWS",
                bound=an.Preceding(),
            )
            .window("w2", base="w1", frame="ROWS", bound=an.Following())
            .select(an.Rank().over(window="w2").as_("r"))
        )

        with self.assertRaises(WindowException) as ctx:
            str(query)
        message = str(ctx.exception)
        self.assertIn("'w2'", message)
        self.assertIn("frame", message)

    def test_valid_child_orderby_when_parent_has_none_is_accepted(self):
        query = (
            Query.from_(self.table)
            .window("w1", partition=[self.table.g])
            .window("w2", base="w1", orderby=[self.table.d])
            .select(an.Rank().over(window="w2").as_("r"))
        )

        self.assertIn('"w2" AS ("w1" ORDER BY "d")', str(query))

    def test_valid_child_frame_when_parent_has_none_is_accepted(self):
        query = (
            Query.from_(self.table)
            .window("w1", partition=[self.table.g], orderby=[self.table.d])
            .window("w2", base="w1", frame="ROWS", bound=an.Preceding())
            .select(an.Rank().over(window="w2").as_("r"))
        )

        self.assertIn('"w2" AS ("w1" ROWS UNBOUNDED PRECEDING)', str(query))

    def test_function_cannot_combine_named_window_with_partition(self):
        with self.assertRaises(WindowException) as ctx:
            an.Rank().over(self.table.g, window="w")

        self.assertIn("PARTITION BY", str(ctx.exception))

    def test_function_cannot_override_named_window_order_by(self):
        query = (
            Query.from_(self.table)
            .window("w", partition=[self.table.g], orderby=[self.table.d])
            .select(an.Rank().over(window="w").orderby(self.table.v).as_("r"))
        )

        with self.assertRaises(WindowException) as ctx:
            str(query)
        message = str(ctx.exception)
        self.assertIn("'w'", message)
        self.assertIn("ORDER BY", message)

    def test_function_cannot_override_named_window_frame(self):
        query = (
            Query.from_(self.table)
            .window(
                "w",
                partition=[self.table.g],
                orderby=[self.table.d],
                frame="ROWS",
                bound=an.Preceding(),
            )
            .select(an.Sum(self.table.v).over(window="w").rows(an.Following()).as_("s"))
        )

        with self.assertRaises(WindowException) as ctx:
            str(query)
        self.assertIn("frame", str(ctx.exception))

    def test_window_requires_bounds_when_frame_mode_given(self):
        with self.assertRaises(WindowException):
            Window("w", partition=[self.table.g], frame="ROWS")

    def test_window_requires_frame_mode_when_bounds_given(self):
        with self.assertRaises(WindowException):
            Window("w", partition=[self.table.g], bound=an.Preceding())


class NamedWindowImmutabilityTests(unittest.TestCase):
    table = Table("events")

    def test_window_definition_returns_new_query_builder(self):
        query_a = Query.from_(self.table).window("w", partition=[self.table.g]).select(self.table.id)
        query_b = query_a.window("w2", base="w", orderby=[self.table.d])

        self.assertIsNot(query_a, query_b)
        self.assertEqual(["w"], [window.name for window in query_a._windows])
        self.assertEqual(["w", "w2"], [window.name for window in query_b._windows])

    def test_named_window_query_is_copyable(self):
        query = (
            Query.from_(self.table)
            .window("w1", partition=[self.table.g])
            .window("w2", base="w1", orderby=[self.table.d])
            .select(an.Rank().over(window="w2").as_("r"))
        )

        for copy_method in (copy, deepcopy):
            with self.subTest(copy=copy_method.__name__):
                cloned = copy_method(query)
                self.assertIsNot(cloned, query)
                self.assertEqual(str(query), str(cloned))

    def test_analytic_function_window_reference_is_copyable(self):
        function = an.Rank().over(window="w")

        for copy_method in (copy, deepcopy):
            with self.subTest(copy=copy_method.__name__):
                cloned = copy_method(function)
                self.assertEqual("w", cloned._window)
                self.assertEqual(function.name, cloned.name)
                self.assertEqual(function.args, cloned.args)

    def test_over_returns_new_function_without_mutating_source(self):
        source = an.Rank()
        referenced = source.over(window="w")

        self.assertIsNone(source._window)
        self.assertEqual("w", referenced._window)
        self.assertIsNot(source, referenced)

    def test_replace_table_updates_window_definitions(self):
        replacement = Table("other")
        query = (
            Query.from_(self.table)
            .window("w", partition=[self.table.g], orderby=[self.table.d])
            .select(an.Rank().over(window="w").as_("r"))
        )

        replaced = query.replace_table(self.table, replacement)

        self.assertEqual(
            'SELECT RANK() OVER "w" "r" FROM "other" WINDOW "w" AS (PARTITION BY "g" ORDER BY "d")',
            str(replaced),
        )
        self.assertEqual(
            'SELECT RANK() OVER "w" "r" FROM "events" WINDOW "w" AS (PARTITION BY "g" ORDER BY "d")',
            str(query),
        )


class NamedWindowSQLiteExecutionTests(unittest.TestCase):
    table = Table("events")

    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("CREATE TABLE events (g INTEGER, d INTEGER, v INTEGER)")
        self.connection.executemany(
            "INSERT INTO events VALUES (?,?,?)",
            [(1, 1, 10), (1, 2, 20), (2, 1, 5), (2, 2, 15), (2, 3, 25)],
        )

    def tearDown(self):
        self.connection.close()

    def _execute(self, query):
        return self.connection.execute(str(query)).fetchall()

    def test_shared_named_window_executes(self):
        query = (
            SQLLiteQuery.from_(self.table)
            .window("w", partition=[self.table.g], orderby=[self.table.d])
            .select(
                self.table.g,
                self.table.d,
                an.Rank().over(window="w").as_("r"),
                an.Sum(self.table.v).over(window="w").as_("running"),
            )
            .orderby(self.table.g, self.table.d)
        )

        self.assertEqual(
            [
                (1, 1, 1, 10),
                (1, 2, 2, 30),
                (2, 1, 1, 5),
                (2, 2, 2, 20),
                (2, 3, 3, 45),
            ],
            self._execute(query),
        )

    def test_two_level_window_inheritance_executes(self):
        query = (
            SQLLiteQuery.from_(self.table)
            .window("w1", partition=[self.table.g])
            .window("w2", base="w1", orderby=[self.table.d])
            .window("w3", base="w2", frame="ROWS", bound=an.Preceding())
            .select(
                self.table.g,
                self.table.d,
                an.Rank().over(window="w2").as_("r"),
                an.Sum(self.table.v).over(window="w3").as_("s"),
            )
            .orderby(self.table.g, self.table.d)
        )

        self.assertEqual(
            [
                (1, 1, 1, 10),
                (1, 2, 2, 30),
                (2, 1, 1, 5),
                (2, 2, 2, 20),
                (2, 3, 3, 45),
            ],
            self._execute(query),
        )

    def test_subquery_window_shadowing_executes(self):
        sub = Table("sub")
        inner = (
            SQLLiteQuery.from_(self.table)
            .window("w", partition=[self.table.g], orderby=[self.table.d])
            .select(an.Rank().over(window="w").as_("r"), self.table.g, self.table.d)
        )
        query = (
            SQLLiteQuery.from_(inner.as_("sub"))
            .window("w", partition=[sub.g], orderby=[sub.r])
            .select(sub.g, sub.r, an.DenseRank().over(window="w").as_("dr"))
            .orderby(sub.g, sub.r)
        )

        self.assertEqual(
            [(1, 1, 1), (1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 3, 3)],
            self._execute(query),
        )

    def test_function_level_orderby_refinement_executes(self):
        query = (
            SQLLiteQuery.from_(self.table)
            .window("w", partition=[self.table.g])
            .select(
                self.table.g,
                self.table.d,
                an.Rank().over(window="w").orderby(self.table.d, order=Order.desc).as_("r"),
            )
            .orderby(self.table.g, self.table.d)
        )

        self.assertEqual(
            [(1, 1, 2), (1, 2, 1), (2, 1, 3), (2, 2, 2), (2, 3, 1)],
            self._execute(query),
        )

    def test_function_level_frame_refinement_executes(self):
        query = (
            SQLLiteQuery.from_(self.table)
            .window("w", partition=[self.table.g], orderby=[self.table.d])
            .select(
                self.table.g,
                self.table.d,
                an.Sum(self.table.v).over(window="w").rows(an.Preceding()).as_("s"),
            )
            .orderby(self.table.g, self.table.d)
        )

        self.assertEqual(
            [(1, 1, 10), (1, 2, 30), (2, 1, 5), (2, 2, 20), (2, 3, 45)],
            self._execute(query),
        )
