import sqlite3
import unittest

from pypika import CustomFunction, Nulls, Order, QmarkParameter, Query, Table, functions as fn
from pypika.dialects import SQLLiteQuery
from pypika.utils import QueryException


class SeekAfterSqliteTests(unittest.TestCase):
    table = Table("events")
    rows = [
        (1, "A", 10, 1),
        (2, "A", 10, 1),
        (3, "A", 5, 1),
        (4, "A", None, 1),
        (5, "A", None, 1),
        (6, "B", 30, 1),
        (7, "B", 20, 1),
        (8, "B", None, 1),
        (9, None, 40, 1),
        (10, None, 40, 1),
        (11, None, None, 1),
        (12, "C", 10, 1),
        (13, "C", 10, 1),
        (14, "C", None, 1),
    ]

    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("CREATE TABLE events (id INTEGER, category TEXT, score INTEGER, active INTEGER)")
        self.connection.executemany("INSERT INTO events VALUES (?,?,?,?)", self.rows)
        self.base = (
            SQLLiteQuery.from_(self.table)
            .select(self.table.id, self.table.category, self.table.score)
            .where(self.table.active == 1)
            .orderby(self.table.category, order=Order.asc, nulls=Nulls.first)
            .orderby(self.table.score, order=Order.desc, nulls=Nulls.last)
            .orderby(self.table.id, order=Order.asc, nulls=Nulls.last, unique=True)
        )

    def tearDown(self):
        self.connection.close()

    def fetch(self, query, parameter=None):
        sql = str(query) if parameter is None else query.get_sql(parameter=parameter)
        args = () if parameter is None else parameter.get_parameters()
        return self.connection.execute(sql, args).fetchall()

    @staticmethod
    def cursor(row):
        return row[1], row[2], row[0]

    def test_pages_cover_full_nullable_mixed_direction_order(self):
        expected = self.fetch(self.base)
        self.assertEqual(
            [
                (9, None, 40),
                (10, None, 40),
                (11, None, None),
                (1, "A", 10),
                (2, "A", 10),
                (3, "A", 5),
                (4, "A", None),
                (5, "A", None),
                (6, "B", 30),
                (7, "B", 20),
                (8, "B", None),
                (12, "C", 10),
                (13, "C", 10),
                (14, "C", None),
            ],
            expected,
        )

        actual = []
        cursor = None
        while True:
            page_query = self.base.limit(3) if cursor is None else self.base.seek_after(*cursor).limit(3)
            page = self.fetch(page_query)
            if not page:
                break
            actual.extend(page)
            cursor = self.cursor(page[-1])

        self.assertEqual(expected, actual)

    def test_cursor_inside_null_group_starts_after_the_cursor(self):
        page = self.fetch(self.base.seek_after("A", None, 5).limit(3))

        self.assertEqual([(6, "B", 30), (7, "B", 20), (8, "B", None)], page)

    def test_cursor_at_first_null_to_non_null_boundary(self):
        page = self.fetch(self.base.seek_after(None, None, 11).limit(3))

        self.assertEqual([(1, "A", 10), (2, "A", 10), (3, "A", 5)], page)

    def test_empty_first_page_and_empty_page_after_last_row(self):
        empty_first = self.base.where(self.table.score > 100).limit(3)
        self.assertEqual([], self.fetch(empty_first))

        last = self.fetch(self.base)[-1]
        self.assertEqual([], self.fetch(self.base.seek_after(*self.cursor(last)).limit(3)))

    def test_parameterized_cursor_values(self):
        parameter = QmarkParameter()
        page_query = self.base.seek_after("A", None, 5).limit(3)

        page = self.connection.execute(page_query.get_sql(parameter=parameter), parameter.get_parameters()).fetchall()

        self.assertEqual([(6, "B", 30), (7, "B", 20), (8, "B", None)], page)
        self.assertEqual([1, "A", "A", 5], parameter.get_parameters())

    def test_seek_does_not_mutate_base_where_or_query_copy(self):
        base_sql = str(self.base)
        base_where = self.base._wheres.get_sql()

        first = self.base.seek_after("A", 5, 3)
        second = self.base.seek_after("B", None, 8)

        self.assertEqual(base_sql, str(self.base))
        self.assertEqual(base_where, self.base._wheres.get_sql())
        self.assertIsNot(first, self.base)
        self.assertIsNot(second, self.base)
        self.assertNotEqual(str(first), str(second))
        self.assertIn('"active"=1', str(first))


class SeekAfterValidationTests(unittest.TestCase):
    table = Table("events")

    def ordered(self, *, first_nulls=Nulls.first, tail_unique=True):
        return (
            Query.from_(self.table)
            .select(self.table.id, self.table.category)
            .orderby(self.table.category, order=Order.asc, nulls=first_nulls)
            .orderby(self.table.id, order=Order.desc, nulls=Nulls.last, unique=tail_unique)
        )

    def test_requires_null_placement_for_every_orderby_term(self):
        query = (
            Query.from_(self.table)
            .select(self.table.id)
            .orderby(self.table.category, order=Order.asc)
            .orderby(self.table.id, order=Order.desc, nulls=Nulls.last, unique=True)
        )

        with self.assertRaisesRegex(QueryException, "must specify nulls"):
            query.seek_after("A", 1)

    def test_requires_cursor_value_for_every_orderby_term(self):
        with self.assertRaisesRegex(QueryException, "expected 2 cursor value"):
            self.ordered().seek_after("A")

    def test_rejects_field_as_cursor_value(self):
        with self.assertRaisesRegex(QueryException, "not fields"):
            self.ordered().seek_after(self.table.category, 1)

    def test_rejects_foreign_orderby_field(self):
        other = Table("other")
        query = (
            Query.from_(self.table)
            .select(self.table.id)
            .orderby(other.id, order=Order.asc, nulls=Nulls.first, unique=True)
        )

        with self.assertRaisesRegex(QueryException, "does not belong to the query source"):
            query.seek_after(1)

    def test_rejects_aggregate_orderby_expression(self):
        query = (
            Query.from_(self.table)
            .select(fn.Sum(self.table.id))
            .orderby(fn.Sum(self.table.id), order=Order.asc, nulls=Nulls.first, unique=True)
        )

        with self.assertRaisesRegex(QueryException, "plain column orderby"):
            query.seek_after(1)

    def test_rejects_function_orderby_expression(self):
        query = (
            Query.from_(self.table)
            .select(self.table.id)
            .orderby(CustomFunction("RANDOM"), order=Order.asc, nulls=Nulls.first, unique=True)
        )

        with self.assertRaisesRegex(QueryException, "plain column orderby"):
            query.seek_after(1)

    def test_rejects_non_unique_tail(self):
        with self.assertRaisesRegex(QueryException, "unique=True"):
            self.ordered(tail_unique=False).seek_after("A", 1)

    def test_rejects_null_unique_tail_cursor(self):
        with self.assertRaisesRegex(QueryException, "cannot be NULL"):
            self.ordered().seek_after("A", None)

    def test_unique_key_can_declare_the_tail(self):
        query = (
            Query.from_(self.table)
            .select(self.table.id)
            .orderby(self.table.category, order=Order.asc, nulls=Nulls.first)
            .orderby(self.table.id, order=Order.asc, nulls=Nulls.last)
        )

        page = query.seek_after("A", 5, unique_key=self.table.id)

        self.assertIn('"id">5', str(page))

    def test_revalidates_before_sql_when_ordering_changes_after_seek(self):
        query = self.ordered().seek_after("A", 1)
        changed = query.orderby(self.table.active, order=Order.desc, nulls=Nulls.last)

        with self.assertRaisesRegex(QueryException, "expected 3 cursor value"):
            changed.get_sql()

    def test_rejects_set_operations(self):
        left = Query.from_(self.table).select(self.table.id)
        right = Query.from_(self.table).select(self.table.id)

        with self.assertRaisesRegex(QueryException, "ordinary SELECT"):
            left.union(right).seek_after(1)


if __name__ == "__main__":
    unittest.main()
