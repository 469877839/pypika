import sqlite3
import unittest

from pypika import (
    NULL,
    Case,
    NullsOrdering,
    Order,
    Parameter,
    QmarkParameter,
    Query,
    QueryException,
    SeekException,
    Table,
)
from pypika.functions import Sum
from pypika.terms import Field, Function


class OrderByNullsTests(unittest.TestCase):
    table_logs = Table("logs")

    def test_orderby_renders_nulls_first_and_last(self):
        q = (
            Query.from_(self.table_logs)
            .select(self.table_logs.id)
            .orderby(self.table_logs.ts, order=Order.asc, nulls=NullsOrdering.first)
            .orderby(self.table_logs.id, order=Order.desc, nulls=NullsOrdering.last)
        )

        self.assertEqual(
            'SELECT "id" FROM "logs" ORDER BY "ts" ASC NULLS FIRST,"id" DESC NULLS LAST',
            str(q),
        )

    def test_orderby_nulls_without_direction(self):
        q = Query.from_(self.table_logs).select(self.table_logs.id).orderby(
            self.table_logs.ts, nulls=NullsOrdering.last
        )

        self.assertEqual('SELECT "id" FROM "logs" ORDER BY "ts" NULLS LAST', str(q))

    def test_orderby_without_nulls_is_unchanged(self):
        q = Query.from_(self.table_logs).select(self.table_logs.id).orderby(self.table_logs.ts, order=Order.desc)

        self.assertEqual('SELECT "id" FROM "logs" ORDER BY "ts" DESC', str(q))

    def test_orderby_rejects_non_nulls_ordering_value(self):
        with self.assertRaises(QueryException):
            Query.from_(self.table_logs).select(self.table_logs.id).orderby(self.table_logs.ts, nulls="last")

    def test_set_operation_orderby_renders_nulls(self):
        logs = self.table_logs
        q = (Query.from_(logs).select(logs.id) + Query.from_(logs).select(logs.id)).orderby(
            logs.id, order=Order.desc, nulls=NullsOrdering.first
        )

        self.assertEqual(
            '(SELECT "id" FROM "logs") UNION (SELECT "id" FROM "logs") ORDER BY "id" DESC NULLS FIRST',
            str(q),
        )


class SeekSqlTests(unittest.TestCase):
    table_logs = Table("logs")

    def base(self):
        return Query.from_(self.table_logs).select(self.table_logs.id, self.table_logs.ts)

    def test_asc_nulls_first_with_value(self):
        q = self.base().orderby(self.table_logs.ts, order=Order.asc, nulls=NullsOrdering.first).seek_after(5)

        self.assertEqual(
            'SELECT "id","ts" FROM "logs" WHERE ("ts" IS NOT NULL AND "ts">5) ORDER BY "ts" ASC NULLS FIRST',
            str(q),
        )

    def test_asc_nulls_first_with_null_cursor(self):
        q = self.base().orderby(self.table_logs.ts, order=Order.asc, nulls=NullsOrdering.first).seek_after(None)

        self.assertEqual(
            'SELECT "id","ts" FROM "logs" WHERE ("ts" IS NOT NULL) ORDER BY "ts" ASC NULLS FIRST',
            str(q),
        )

    def test_asc_nulls_last_with_value(self):
        q = self.base().orderby(self.table_logs.ts, order=Order.asc, nulls=NullsOrdering.last).seek_after(5)

        self.assertEqual(
            'SELECT "id","ts" FROM "logs" WHERE (("ts" IS NOT NULL AND "ts">5) OR "ts" IS NULL) '
            'ORDER BY "ts" ASC NULLS LAST',
            str(q),
        )

    def test_asc_nulls_last_with_null_cursor(self):
        q = self.base().orderby(self.table_logs.ts, order=Order.asc, nulls=NullsOrdering.last).seek_after(NULL)

        self.assertEqual(
            'SELECT "id","ts" FROM "logs" WHERE (1=0) ORDER BY "ts" ASC NULLS LAST',
            str(q),
        )

    def test_desc_nulls_first_with_value(self):
        q = self.base().orderby(self.table_logs.ts, order=Order.desc, nulls=NullsOrdering.first).seek_after(5)

        self.assertEqual(
            'SELECT "id","ts" FROM "logs" WHERE ("ts" IS NOT NULL AND "ts"<5) ORDER BY "ts" DESC NULLS FIRST',
            str(q),
        )

    def test_desc_nulls_first_with_null_cursor(self):
        q = self.base().orderby(self.table_logs.ts, order=Order.desc, nulls=NullsOrdering.first).seek_after(None)

        self.assertEqual(
            'SELECT "id","ts" FROM "logs" WHERE ("ts" IS NOT NULL) ORDER BY "ts" DESC NULLS FIRST',
            str(q),
        )

    def test_desc_nulls_last_with_value(self):
        q = self.base().orderby(self.table_logs.ts, order=Order.desc, nulls=NullsOrdering.last).seek_after(5)

        self.assertEqual(
            'SELECT "id","ts" FROM "logs" WHERE (("ts" IS NOT NULL AND "ts"<5) OR "ts" IS NULL) '
            'ORDER BY "ts" DESC NULLS LAST',
            str(q),
        )

    def test_desc_nulls_last_with_null_cursor(self):
        q = self.base().orderby(self.table_logs.ts, order=Order.desc, nulls=NullsOrdering.last).seek_after(None)

        self.assertEqual(
            'SELECT "id","ts" FROM "logs" WHERE (1=0) ORDER BY "ts" DESC NULLS LAST',
            str(q),
        )

    def test_multi_key_seek_builds_or_chain(self):
        q = (
            self.base()
            .orderby(self.table_logs.ts, order=Order.asc, nulls=NullsOrdering.last)
            .orderby(self.table_logs.id, order=Order.desc, nulls=NullsOrdering.first)
            .seek_after(20, 4)
        )

        self.assertEqual(
            'SELECT "id","ts" FROM "logs" '
            'WHERE (("ts" IS NOT NULL AND "ts">20) OR "ts" IS NULL) '
            'OR ("id" IS NOT NULL AND "id"<4 AND "ts"=20) '
            'ORDER BY "ts" ASC NULLS LAST,"id" DESC NULLS FIRST',
            str(q),
        )

    def test_multi_key_seek_with_null_tie_break_uses_is_null(self):
        q = (
            self.base()
            .orderby(self.table_logs.ts, order=Order.asc, nulls=NullsOrdering.last)
            .orderby(self.table_logs.id, order=Order.asc, nulls=NullsOrdering.first)
            .seek_after(None, 3)
        )

        self.assertEqual(
            'SELECT "id","ts" FROM "logs" '
            'WHERE (1=0) OR ("id" IS NOT NULL AND "id">3 AND "ts" IS NULL) '
            'ORDER BY "ts" ASC NULLS LAST,"id" ASC NULLS FIRST',
            str(q),
        )

    def test_seek_after_keeps_existing_where(self):
        q = (
            self.base()
            .where(self.table_logs.id > 0)
            .orderby(self.table_logs.ts, order=Order.desc, nulls=NullsOrdering.last)
            .seek_after(Parameter("?"))
        )

        self.assertEqual(
            'SELECT "id","ts" FROM "logs" WHERE "id">0 AND (("ts" IS NOT NULL AND "ts"<?) OR "ts" IS NULL) '
            'ORDER BY "ts" DESC NULLS LAST',
            str(q),
        )

    def test_seek_after_does_not_mutate_the_original_query(self):
        original = self.base().orderby(
            self.table_logs.id, order=Order.asc, nulls=NullsOrdering.last
        )
        paged = original.seek_after(3)

        self.assertIsNot(original, paged)
        self.assertEqual('SELECT "id","ts" FROM "logs" ORDER BY "id" ASC NULLS LAST', str(original))
        self.assertIn("WHERE", str(paged))

    def test_repeated_seek_after_calls_stack_predicates(self):
        ordered = self.base().orderby(self.table_logs.id, order=Order.asc, nulls=NullsOrdering.last)
        once = ordered.seek_after(1)
        twice = once.seek_after(2)

        self.assertNotIn("WHERE", str(ordered))
        self.assertIn('"id">1', str(once))
        self.assertIn('"id">1', str(twice))
        self.assertIn('"id">2', str(twice))


class SeekValidationTests(unittest.TestCase):
    table_logs = Table("logs")

    def base(self):
        return Query.from_(self.table_logs).select(self.table_logs.id, self.table_logs.ts)

    def test_seek_without_orderby_is_rejected(self):
        with self.assertRaises(SeekException):
            self.base().seek_after(1)

    def test_seek_with_too_few_cursor_values_is_rejected(self):
        q = (
            self.base()
            .orderby(self.table_logs.ts, order=Order.asc, nulls=NullsOrdering.last)
            .orderby(self.table_logs.id, order=Order.asc, nulls=NullsOrdering.last)
        )

        with self.assertRaises(SeekException):
            q.seek_after(1)

    def test_seek_with_too_many_cursor_values_is_rejected(self):
        q = self.base().orderby(self.table_logs.id, order=Order.asc, nulls=NullsOrdering.last)

        with self.assertRaises(SeekException):
            q.seek_after(1, 2)

    def test_seek_without_direction_is_rejected(self):
        q = self.base().orderby(self.table_logs.id, nulls=NullsOrdering.last)

        with self.assertRaises(SeekException):
            q.seek_after(1)

    def test_seek_without_null_position_is_rejected(self):
        q = self.base().orderby(self.table_logs.id, order=Order.asc)

        with self.assertRaises(SeekException):
            q.seek_after(1)

    def test_seek_with_aggregate_orderby_is_rejected(self):
        q = (
            self.base()
            .groupby(self.table_logs.id)
            .orderby(Sum(self.table_logs.ts), order=Order.asc, nulls=NullsOrdering.last)
        )

        with self.assertRaises(SeekException):
            q.seek_after(1)

    def test_seek_with_random_orderby_is_rejected(self):
        q = self.base().orderby(
            Function("RANDOM"), order=Order.asc, nulls=NullsOrdering.last
        )

        with self.assertRaises(SeekException):
            q.seek_after(1)

    def test_seek_with_function_orderby_is_rejected(self):
        q = self.base().orderby(
            Function("LOWER", self.table_logs.id), order=Order.asc, nulls=NullsOrdering.last
        )

        with self.assertRaises(SeekException):
            q.seek_after("a")

    def test_seek_with_case_orderby_is_rejected(self):
        case = Case().when(self.table_logs.ts > 1, "a").else_("b")
        q = self.base().orderby(case, order=Order.asc, nulls=NullsOrdering.last)

        with self.assertRaises(SeekException):
            q.seek_after("a")

    def test_seek_with_field_cursor_value_is_rejected(self):
        q = self.base().orderby(self.table_logs.id, order=Order.asc, nulls=NullsOrdering.last)

        with self.assertRaises(SeekException):
            q.seek_after(Field("other"))

    def test_validation_happens_before_sql_generation(self):
        q = self.base().orderby(Sum(self.table_logs.ts), order=Order.asc, nulls=NullsOrdering.last)

        with self.assertRaises(SeekException):
            q.seek_after(1).get_sql()


class SeekPaginationSqliteTests(unittest.TestCase):
    """
    Paginates a small table with duplicate values and NULLs using seek_after and
    stitches the pages back together, comparing against a single full sort.
    """

    table_logs = Table("logs")

    ROWS = [
        (1, "info", 10),
        (2, "info", 20),
        (3, "warn", None),
        (4, "info", 20),
        (5, "error", 30),
        (6, "warn", None),
        (7, "info", 10),
        (8, "error", None),
        (9, "warn", 40),
        (10, "info", 20),
        (11, "error", 30),
        (12, "info", None),
        (13, "warn", 30),
        (14, "info", 10),
    ]

    SELECTS = [table_logs.id, table_logs.level, table_logs.ts]
    COLUMN_INDEX = {field.name: index for index, field in enumerate(SELECTS)}

    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.addCleanup(self.connection.close)
        self.connection.execute("CREATE TABLE logs (id INTEGER PRIMARY KEY, level TEXT, ts INTEGER)")
        self.connection.executemany("INSERT INTO logs VALUES (?,?,?)", self.ROWS)

    def base_query(self, order_specs):
        query = Query.from_(self.table_logs).select(*self.SELECTS)
        for field, order, nulls in order_specs:
            query = query.orderby(field, order=order, nulls=nulls)
        return query

    def paginate(self, order_specs, page_size, where=None):
        query = self.base_query(order_specs)
        if where is not None:
            query = query.where(where)

        cursor = None
        pages = []
        while True:
            page_query = query.limit(page_size)
            if cursor is not None:
                page_query = page_query.seek_after(*cursor)
            page = self.connection.execute(str(page_query)).fetchall()
            pages.append(page)
            if len(page) < page_size:
                return pages
            last_row = page[-1]
            cursor = tuple(last_row[self.COLUMN_INDEX[field.name]] for field, _, _ in order_specs)

    def reference(self, order_specs, where=None):
        orderby = ",".join(
            '"{field}" {order} {nulls}'.format(field=field.name, order=order.value, nulls=nulls.value)
            for field, order, nulls in order_specs
        )
        sql = 'SELECT id, level, ts FROM logs'
        if where is not None:
            sql += ' WHERE ' + where
        sql += ' ORDER BY ' + orderby
        return self.connection.execute(sql).fetchall()

    def assert_pagination_matches_full_sort(self, order_specs, page_size, where=None, where_sql=None):
        pages = self.paginate(order_specs, page_size, where=where)
        stitched = [row for page in pages for row in page]
        expected = self.reference(order_specs, where=where_sql)

        self.assertEqual(expected, stitched)
        self.assertEqual(len(expected), len(stitched))
        self.assertEqual(len(set(stitched)), len(stitched))
        return pages

    def test_asc_nulls_last_matches_full_sort(self):
        order_specs = [
            (self.table_logs.ts, Order.asc, NullsOrdering.last),
            (self.table_logs.id, Order.asc, NullsOrdering.last),
        ]
        for page_size in (1, 2, 3, 4, 5, 7):
            with self.subTest(page_size=page_size):
                self.assert_pagination_matches_full_sort(order_specs, page_size)

    def test_desc_nulls_first_matches_full_sort(self):
        order_specs = [
            (self.table_logs.ts, Order.desc, NullsOrdering.first),
            (self.table_logs.id, Order.asc, NullsOrdering.last),
        ]
        for page_size in (1, 2, 3, 4, 5, 7):
            with self.subTest(page_size=page_size):
                self.assert_pagination_matches_full_sort(order_specs, page_size)

    def test_desc_nulls_last_matches_full_sort(self):
        order_specs = [
            (self.table_logs.ts, Order.desc, NullsOrdering.last),
            (self.table_logs.id, Order.desc, NullsOrdering.last),
        ]
        for page_size in (1, 2, 3, 4, 5, 7):
            with self.subTest(page_size=page_size):
                self.assert_pagination_matches_full_sort(order_specs, page_size)

    def test_asc_nulls_first_matches_full_sort(self):
        order_specs = [
            (self.table_logs.ts, Order.asc, NullsOrdering.first),
            (self.table_logs.id, Order.asc, NullsOrdering.first),
        ]
        for page_size in (1, 2, 3, 4, 5, 7):
            with self.subTest(page_size=page_size):
                self.assert_pagination_matches_full_sort(order_specs, page_size)

    def test_reverse_field_ordering_matches_full_sort(self):
        # descending on a text column with duplicates, unique descending tail key
        order_specs = [
            (self.table_logs.level, Order.desc, NullsOrdering.last),
            (self.table_logs.id, Order.desc, NullsOrdering.last),
        ]
        for page_size in (1, 2, 3, 4, 5, 7):
            with self.subTest(page_size=page_size):
                self.assert_pagination_matches_full_sort(order_specs, page_size)

    def test_mixed_directions_three_keys_matches_full_sort(self):
        order_specs = [
            (self.table_logs.level, Order.asc, NullsOrdering.last),
            (self.table_logs.ts, Order.desc, NullsOrdering.first),
            (self.table_logs.id, Order.asc, NullsOrdering.last),
        ]
        for page_size in (1, 2, 3, 4, 5, 7):
            with self.subTest(page_size=page_size):
                self.assert_pagination_matches_full_sort(order_specs, page_size)

    def test_mixed_null_positions_matches_full_sort(self):
        order_specs = [
            (self.table_logs.level, Order.asc, NullsOrdering.first),
            (self.table_logs.ts, Order.asc, NullsOrdering.last),
            (self.table_logs.id, Order.desc, NullsOrdering.last),
        ]
        for page_size in (1, 2, 3, 4, 5, 7):
            with self.subTest(page_size=page_size):
                self.assert_pagination_matches_full_sort(order_specs, page_size)

    def test_empty_first_page(self):
        order_specs = [
            (self.table_logs.ts, Order.asc, NullsOrdering.last),
            (self.table_logs.id, Order.asc, NullsOrdering.last),
        ]
        pages = self.paginate(order_specs, 4, where=self.table_logs.level == "debug")

        self.assertEqual([[]], pages)

    def test_cursor_at_null_boundary_asc_nulls_last(self):
        # the cursor is the last non-NULL row; the next page must start with the NULL group
        order_specs = [
            (self.table_logs.ts, Order.asc, NullsOrdering.last),
            (self.table_logs.id, Order.asc, NullsOrdering.last),
        ]
        query = self.base_query(order_specs).limit(4).seek_after(40, 9)
        page = self.connection.execute(str(query)).fetchall()

        self.assertEqual([(3, "warn", None), (6, "warn", None), (8, "error", None), (12, "info", None)], page)

    def test_cursor_at_null_boundary_asc_nulls_first(self):
        # the cursor is the last NULL row; the next page must start with non-NULL values
        order_specs = [
            (self.table_logs.ts, Order.asc, NullsOrdering.first),
            (self.table_logs.id, Order.asc, NullsOrdering.first),
        ]
        query = self.base_query(order_specs).limit(4).seek_after(None, 12)
        page = self.connection.execute(str(query)).fetchall()

        self.assertEqual([(1, "info", 10), (7, "info", 10), (14, "info", 10), (2, "info", 20)], page)

    def test_cursor_at_null_boundary_desc_nulls_first(self):
        order_specs = [
            (self.table_logs.ts, Order.desc, NullsOrdering.first),
            (self.table_logs.id, Order.asc, NullsOrdering.first),
        ]
        query = self.base_query(order_specs).limit(4).seek_after(None, 12)
        page = self.connection.execute(str(query)).fetchall()

        self.assertEqual([(9, "warn", 40), (5, "error", 30), (11, "error", 30), (13, "warn", 30)], page)

    def test_cursor_at_null_boundary_desc_nulls_last(self):
        order_specs = [
            (self.table_logs.ts, Order.desc, NullsOrdering.last),
            (self.table_logs.id, Order.asc, NullsOrdering.last),
        ]
        query = self.base_query(order_specs).limit(4).seek_after(10, 7)
        page = self.connection.execute(str(query)).fetchall()

        self.assertEqual(
            [(14, "info", 10), (3, "warn", None), (6, "warn", None), (8, "error", None)], page
        )

    def test_cursor_at_last_null_row_returns_empty_page(self):
        order_specs = [
            (self.table_logs.ts, Order.asc, NullsOrdering.last),
            (self.table_logs.id, Order.asc, NullsOrdering.last),
        ]
        query = self.base_query(order_specs).limit(4).seek_after(None, 12)
        page = self.connection.execute(str(query)).fetchall()

        self.assertEqual([], page)

    def test_pagination_with_existing_where_matches_filtered_sort(self):
        order_specs = [
            (self.table_logs.ts, Order.asc, NullsOrdering.last),
            (self.table_logs.id, Order.asc, NullsOrdering.last),
        ]
        pages = self.assert_pagination_matches_full_sort(
            order_specs,
            3,
            where=self.table_logs.level != "info",
            where_sql='"level"<>"info"',
        )
        stitched = [row for page in pages for row in page]
        self.assertEqual(7, len(stitched))
        self.assertTrue(all(row[1] != "info" for row in stitched))

    def test_parameterized_cursor_executes_in_sqlite(self):
        order_specs = [
            (self.table_logs.ts, Order.asc, NullsOrdering.last),
            (self.table_logs.id, Order.asc, NullsOrdering.last),
        ]
        query = self.base_query(order_specs).limit(4).seek_after(Parameter("?"), Parameter("?"))

        # positional placeholders cannot be referenced twice, so the leading key's
        # value is bound once per occurrence in the OR chain
        page = self.connection.execute(str(query), (20, 4, 20)).fetchall()

        self.assertEqual([(10, "info", 20), (5, "error", 30), (11, "error", 30), (13, "warn", 30)], page)

    def test_collected_parameter_cursor_executes_in_sqlite(self):
        order_specs = [
            (self.table_logs.ts, Order.asc, NullsOrdering.last),
            (self.table_logs.id, Order.asc, NullsOrdering.last),
        ]
        query = self.base_query(order_specs).limit(4).seek_after(20, 4)

        collector = QmarkParameter()
        sql = query.get_sql(parameter=collector)
        page = self.connection.execute(sql, collector.get_parameters()).fetchall()

        self.assertEqual([20, 4, 20], collector.get_parameters())
        self.assertEqual([(10, "info", 20), (5, "error", 30), (11, "error", 30), (13, "warn", 30)], page)

    def test_named_parameter_cursor_executes_in_sqlite(self):
        order_specs = [
            (self.table_logs.ts, Order.asc, NullsOrdering.last),
            (self.table_logs.id, Order.asc, NullsOrdering.last),
        ]
        query = (
            self.base_query(order_specs)
            .limit(4)
            .seek_after(Parameter(":after_ts"), Parameter(":after_id"))
        )
        page = self.connection.execute(str(query), {"after_ts": 20, "after_id": 4}).fetchall()

        self.assertEqual([(10, "info", 20), (5, "error", 30), (11, "error", 30), (13, "warn", 30)], page)
