import sqlite3
import unittest

from pypika import MSSQLQuery, OracleQuery, Order, Query, Table
from pypika import analytics as an
from pypika.terms import Window
from pypika.utils import WindowException


class NamedWindowTests(unittest.TestCase):
    table = Table("ledger")

    def setUp(self):
        self.table = Table("ledger")

    def test_multiple_analytic_functions_share_one_window(self):
        window = Window("w").partitionby(self.table.account).orderby(self.table.created_at)
        query = (
            Query.from_(self.table)
            .select(
                an.Sum(self.table.amount).over(window).as_("running_total"),
                an.Avg(self.table.amount).over(window).as_("running_avg"),
            )
            .window(window)
        )

        self.assertEqual(
            'SELECT SUM("amount") OVER "w" "running_total",'
            'AVG("amount") OVER "w" "running_avg" '
            'FROM "ledger" '
            'WINDOW "w" AS (PARTITION BY "account" ORDER BY "created_at")',
            str(query),
        )

    def test_two_level_window_inheritance(self):
        base = Window("base").partitionby(self.table.account)
        ordered = Window("ordered").based_on(base).orderby(self.table.created_at)
        rolling = Window("rolling").based_on(ordered).rows(an.Preceding(2), an.CURRENT_ROW)
        query = (
            Query.from_(self.table)
            .select(
                self.table.created_at,
                an.Sum(self.table.amount).over(rolling).as_("rolling_total"),
            )
            .window(base, ordered, rolling)
        )

        self.assertEqual(
            'SELECT "created_at",SUM("amount") OVER "rolling" "rolling_total" '
            'FROM "ledger" '
            'WINDOW "base" AS (PARTITION BY "account"),'
            '"ordered" AS ("base" ORDER BY "created_at"),'
            '"rolling" AS ("ordered" ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)',
            str(query),
        )

    def test_window_definition_can_be_declared_before_its_base(self):
        child = Window("child").based_on("parent").orderby(self.table.created_at)
        parent = Window("parent").partitionby(self.table.account)
        query = Query.from_(self.table).select(an.Sum(self.table.amount).over(child)).window(child, parent)

        self.assertEqual(
            'SELECT SUM("amount") OVER "child" '
            'FROM "ledger" '
            'WINDOW "parent" AS (PARTITION BY "account"),'
            '"child" AS ("parent" ORDER BY "created_at")',
            str(query),
        )

    def test_subquery_can_shadow_outer_window_name(self):
        inner_window = Window("w").orderby(self.table.created_at)
        inner = (
            Query.from_(self.table)
            .select(self.table.id, an.RowNumber().over(inner_window).as_("rn"))
            .window(inner_window)
        )
        outer_window = Window("w").partitionby(inner.id)
        outer = (
            Query.from_(inner)
            .select(inner.id, an.Sum(inner.id).over(outer_window).as_("grouped"))
            .window(outer_window)
        )

        sql = str(outer)
        self.assertEqual(2, sql.count('WINDOW "w" AS'))
        self.assertIn('ROW_NUMBER() OVER "w"', sql)
        self.assertIn('SUM("sq0"."id") OVER "w"', sql)

    def test_duplicate_window_name_is_rejected(self):
        query = Query.from_(self.table).window(Window("w"), Window("w"))
        with self.assertRaisesRegex(WindowException, "Duplicate window name 'w'"):
            query.get_sql()

    def test_outer_window_name_is_not_visible_in_a_subquery(self):
        inner = Query.from_(self.table).select(an.Sum(self.table.amount).over("w"))
        outer = (
            Query.from_(inner)
            .select(inner.amount)
            .window(Window("w").partitionby(self.table.account))
        )

        with self.assertRaisesRegex(WindowException, "Unknown window 'w' in this SELECT"):
            outer.get_sql()

    def test_unknown_string_window_reference_is_rejected(self):
        query = (
            Query.from_(self.table)
            .select(an.Sum(self.table.amount).over("missing"))
            .window(Window("defined").partitionby(self.table.account))
        )

        with self.assertRaisesRegex(WindowException, "Unknown window 'missing'"):
            query.get_sql()

    def test_unknown_window_reference_is_rejected(self):
        query = (
            Query.from_(self.table)
            .select(an.Sum(self.table.amount).over(Window("missing")))
            .window(Window("defined").partitionby(self.table.account))
        )

        with self.assertRaisesRegex(WindowException, "Unknown window 'missing'"):
            query.get_sql()

    def test_unknown_base_window_is_rejected(self):
        query = Query.from_(self.table).window(Window("child").based_on("missing"))

        with self.assertRaisesRegex(WindowException, "Window 'child' references unknown window 'missing'"):
            query.get_sql()

    def test_circular_inheritance_is_rejected(self):
        a = Window("a").based_on("b")
        b = Window("b").based_on("a")
        query = Query.from_(self.table).select(an.Sum(self.table.amount).over(a)).window(a, b)

        with self.assertRaisesRegex(WindowException, r"Circular window inheritance: a -> b -> a"):
            query.get_sql()

    def test_inheritance_cannot_replace_partition(self):
        parent = Window("parent").partitionby(self.table.account)
        child = Window("child").based_on(parent).partitionby(self.table.region)
        query = Query.from_(self.table).window(parent, child)

        with self.assertRaisesRegex(WindowException, "Window 'child' cannot specify PARTITION BY"):
            query.get_sql()

    def test_inheritance_cannot_insert_partition_before_order(self):
        parent = Window("parent").orderby(self.table.created_at)
        child = Window("child").based_on(parent).partitionby(self.table.account)
        query = Query.from_(self.table).window(parent, child)

        with self.assertRaisesRegex(WindowException, "Window 'child' cannot specify PARTITION BY"):
            query.get_sql()

    def test_inheritance_cannot_replace_order(self):
        parent = Window("parent").orderby(self.table.created_at)
        child = Window("child").based_on(parent).orderby(self.table.id, order=Order.desc)
        query = Query.from_(self.table).window(parent, child)

        with self.assertRaisesRegex(WindowException, "Window 'child' cannot specify ORDER BY"):
            query.get_sql()

    def test_inheritance_cannot_insert_order_before_frame(self):
        parent = Window("parent").rows(an.Preceding(1))
        child = Window("child").based_on(parent).orderby(self.table.created_at)
        query = Query.from_(self.table).window(parent, child)

        with self.assertRaisesRegex(WindowException, "Window 'child' cannot specify ORDER BY"):
            query.get_sql()

    def test_inheritance_cannot_replace_frame(self):
        parent = Window("parent").rows(an.Preceding(1))
        child = Window("child").based_on(parent).rows(an.Preceding(2))
        query = Query.from_(self.table).window(parent, child)

        with self.assertRaisesRegex(WindowException, "Window 'child' cannot specify a frame"):
            query.get_sql()

    def test_inheritance_can_append_clauses_in_sql_order(self):
        parent = Window("parent").partitionby(self.table.account)
        child = Window("child").based_on(parent).orderby(self.table.created_at).rows(an.Preceding(1))
        query = Query.from_(self.table).select(an.Sum(self.table.amount).over(child)).window(parent, child)

        self.assertEqual(
            'SELECT SUM("amount") OVER "child" '
            'FROM "ledger" '
            'WINDOW "parent" AS (PARTITION BY "account"),'
            '"child" AS ("parent" ORDER BY "created_at" ROWS 1 PRECEDING)',
            str(query),
        )

    def test_analytic_cannot_override_named_window_clause(self):
        window = Window("w").partitionby(self.table.account)
        query = Query.from_(self.table).select(an.Sum(self.table.amount).over(window).over(self.table.region))

        with self.assertRaisesRegex(WindowException, "cannot add PARTITION BY to window 'w'"):
            query.window(window).get_sql()

    def test_copying_windows_and_analytic_functions_does_not_mutate_originals(self):
        window = Window("w").partitionby(self.table.account)
        ordered = window.orderby(self.table.created_at)
        function = an.Sum(self.table.amount).over(window)
        ordered_function = function.orderby(self.table.id)

        self.assertEqual('PARTITION BY "account"', window.get_definition_sql(quote_char='"'))
        self.assertEqual(
            'PARTITION BY "account" ORDER BY "created_at"', ordered.get_definition_sql(quote_char='"')
        )
        self.assertEqual('SUM("amount") OVER "w"', function.get_sql(quote_char='"'))
        self.assertEqual(
            'SUM("amount") OVER ("w" ORDER BY "id")', ordered_function.get_sql(quote_char='"')
        )
        self.assertEqual(self.table.id, ordered_function._orderbys[0][0])

    def test_query_builder_copy_keeps_window_definitions(self):
        window = Window("w").partitionby(self.table.account)
        base = Query.from_(self.table).select(an.Sum(self.table.amount).over(window)).window(window)
        changed = base.orderby(self.table.id)

        self.assertNotIn("ORDER BY", str(base))
        self.assertTrue(str(changed).endswith('ORDER BY "id"'))

    def test_dialects_without_native_window_clause_expand_definitions(self):
        base = Window("base").partitionby(self.table.account)
        child = Window("child").based_on(base).orderby(self.table.created_at)
        for query_cls in (MSSQLQuery, OracleQuery):
            with self.subTest(query_cls=query_cls):
                sql = str(
                    query_cls.from_(self.table)
                    .select(an.Sum(self.table.amount).over(child))
                    .window(base, child)
                )
                self.assertNotIn("WINDOW", sql)
                self.assertNotIn('OVER "child"', sql)
                self.assertIn("PARTITION BY", sql)
                self.assertIn("ORDER BY", sql)


class NamedWindowSQLiteTests(unittest.TestCase):
    def setUp(self):
        self.table = Table("ledger")
        self.connection = sqlite3.connect(":memory:")
        self.connection.executescript(
            """
            CREATE TABLE ledger (account TEXT, id INTEGER, amount INTEGER);
            INSERT INTO ledger VALUES
                ('a', 1, 10),
                ('a', 2, 20),
                ('a', 3, 30),
                ('b', 1, 5);
            """
        )

    def tearDown(self):
        self.connection.close()

    def test_shared_window_and_two_level_inheritance_execute(self):
        base = Window("base").partitionby(self.table.account).orderby(self.table.id)
        rolling = Window("rolling").based_on(base).rows(an.Preceding(1), an.CURRENT_ROW)
        query = (
            Query.from_(self.table)
            .select(
                self.table.id,
                an.Sum(self.table.amount).over(base).as_("total"),
                an.Sum(self.table.amount).over(rolling).as_("rolling_total"),
            )
            .window(base, rolling)
        )

        rows = self.connection.execute(str(query)).fetchall()

        self.assertEqual([(1, 10, 10), (2, 30, 30), (3, 60, 50), (1, 5, 5)], rows)

    def test_shadowed_subquery_window_executes(self):
        inner_window = Window("w").partitionby(self.table.account).orderby(self.table.id)
        inner = (
            Query.from_(self.table)
            .select(self.table.account, self.table.id, an.RowNumber().over(inner_window).as_("rn"))
            .window(inner_window)
        )
        outer_window = Window("w").partitionby(inner.account)
        outer = (
            Query.from_(inner)
            .select(inner.id, an.Sum(inner.id).over(outer_window).as_("account_total"))
            .window(outer_window)
        )

        rows = self.connection.execute(str(outer)).fetchall()

        self.assertEqual([(1, 6), (2, 6), (3, 6), (1, 1)], rows)
