import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from pypika import (
    FormatParameter,
    NamedParameter,
    NumericParameter,
    Parameter,
    ParameterException,
    PyformatParameter,
    QmarkParameter,
    Query,
    Table,
    Tables,
)
from pypika.dialects import ClickHouseQuery, MSSQLQuery, MySQLQuery, PostgreSQLQuery, SQLLiteQuery
from pypika.terms import ListParameter, ParameterValueWrapper, Term


class ExplodingTerm(Term):
    """A term which always fails while rendering, used to abort a compilation mid-walk."""

    def get_sql(self, **kwargs):
        raise RuntimeError("boom")


class ParameterizedCompileTests(unittest.TestCase):
    table_abc, table_efg = Tables("abc", "efg")

    def test_default_collector_is_qmark(self):
        q = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")

        sql, parameters = q.get_parameterized_sql()

        self.assertEqual('SELECT * FROM "abc" WHERE "category"=?', sql)
        self.assertEqual(["foobar"], parameters)

    def test_all_paramstyles(self):
        q = Query.into(self.table_abc).columns("a", "b").insert(1, "foo")

        self.assertEqual(
            ('INSERT INTO "abc" ("a","b") VALUES (?,?)', [1, "foo"]),
            q.get_parameterized_sql(QmarkParameter),
        )
        self.assertEqual(
            ('INSERT INTO "abc" ("a","b") VALUES (%s,%s)', [1, "foo"]),
            q.get_parameterized_sql(FormatParameter),
        )
        self.assertEqual(
            ('INSERT INTO "abc" ("a","b") VALUES (:1,:2)', [1, "foo"]),
            q.get_parameterized_sql(NumericParameter),
        )
        self.assertEqual(
            (
                'INSERT INTO "abc" ("a","b") VALUES (:param1,:param2)',
                {"param1": 1, "param2": "foo"},
            ),
            q.get_parameterized_sql(NamedParameter),
        )
        self.assertEqual(
            (
                'INSERT INTO "abc" ("a","b") VALUES (%(param1)s,%(param2)s)',
                {"param1": 1, "param2": "foo"},
            ),
            q.get_parameterized_sql(PyformatParameter),
        )

    def test_instance_style_is_copied_without_its_state(self):
        q = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")
        style = ListParameter(placeholder=lambda idx: f"&{idx + 1}")

        sql_first, params_first = q.get_parameterized_sql(style)
        sql_second, params_second = q.get_parameterized_sql(style)

        self.assertEqual('SELECT * FROM "abc" WHERE "category"=&1', sql_first)
        self.assertEqual(["foobar"], params_first)
        # The same style instance can be reused: collection state never leaks
        # between compilations and the instance itself stays untouched.
        self.assertEqual(sql_first, sql_second)
        self.assertEqual(params_first, params_second)
        self.assertEqual([], style.get_parameters())

    def test_repeated_compilation_does_not_accumulate(self):
        q = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")

        first_sql, first_params = q.get_parameterized_sql()
        second_sql, second_params = q.get_parameterized_sql()

        self.assertEqual('SELECT * FROM "abc" WHERE "category"=?', first_sql)
        self.assertEqual(["foobar"], first_params)
        self.assertEqual(first_sql, second_sql)
        self.assertEqual(first_params, second_params)

    def test_shared_expression_in_select_and_where_collects_each_placeholder(self):
        expression = self.table_abc.score * 2
        q = Query.from_(self.table_abc).select(expression).where(expression > 10)

        sql, parameters = q.get_parameterized_sql()

        self.assertEqual('SELECT "score"*? FROM "abc" WHERE "score"*?>?', sql)
        # Every rendered placeholder gets a value, even though the same term
        # object appears twice in the tree.
        self.assertEqual([2, 2, 10], parameters)

    def test_shared_expression_named_placeholders_are_unique(self):
        expression = self.table_abc.score * 2
        q = Query.from_(self.table_abc).select(expression).where(expression > 10)

        sql, parameters = q.get_parameterized_sql(NamedParameter)

        self.assertEqual('SELECT "score"*:param1 FROM "abc" WHERE "score"*:param2>:param3', sql)
        self.assertEqual({"param1": 2, "param2": 2, "param3": 10}, parameters)

    def test_concurrent_compilations_are_isolated(self):
        q = (
            Query.from_(self.table_abc)
            .select("*")
            .where((self.table_abc.category == "foobar") & (self.table_abc.age >= 18))
        )

        def compile_once(_):
            return q.get_parameterized_sql()

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(compile_once, range(50)))

        expected = ('SELECT * FROM "abc" WHERE "category"=? AND "age">=?', ["foobar", 18])
        self.assertTrue(all(result == expected for result in results))

    def test_failed_compilation_does_not_poison_retry(self):
        good_query = Query.from_(self.table_abc).select("*").where(self.table_abc.name == "ok")
        broken_query = Query.from_(self.table_abc).select(ExplodingTerm()).where(self.table_abc.name == "partial")

        for _ in range(3):
            with self.assertRaises(RuntimeError):
                broken_query.get_parameterized_sql()

        sql, parameters = good_query.get_parameterized_sql()
        self.assertEqual('SELECT * FROM "abc" WHERE "name"=?', sql)
        self.assertEqual(["ok"], parameters)

    def test_manual_parameter_without_binding_raises_in_strict_mode(self):
        q = (
            Query.from_(self.table_abc)
            .select("*")
            .where((self.table_abc.id == Parameter(":id")) & (self.table_abc.name == "foobar"))
        )

        with self.assertRaises(ParameterException) as ctx:
            q.get_parameterized_sql(NamedParameter)

        self.assertIn(":id", str(ctx.exception))

    def test_manual_parameter_can_be_bound_outside_pypika_in_non_strict_mode(self):
        q = (
            Query.from_(self.table_abc)
            .select("*")
            .where((self.table_abc.id == Parameter(":id")) & (self.table_abc.name == "foobar"))
        )

        sql, parameters = q.get_parameterized_sql(NamedParameter, strict=False)

        # The hand-written placeholder is preserved and only the auto-wrapped
        # value is collected: PyPika does not invent or swallow a binding for :id.
        self.assertEqual('SELECT * FROM "abc" WHERE "id"=:id AND "name"=:param1', sql)
        self.assertEqual({"param1": "foobar"}, parameters)

    def test_parameter_value_wrapper_binds_manual_placeholder(self):
        q = (
            Query.from_(self.table_abc)
            .select("*")
            .where(
                (self.table_abc.id == ParameterValueWrapper(NamedParameter("id"), 7))
                & (self.table_abc.name == "foobar")
            )
        )

        sql, parameters = q.get_parameterized_sql(NamedParameter)

        self.assertEqual('SELECT * FROM "abc" WHERE "id"=:id AND "name"=:param2', sql)
        self.assertEqual({"id": 7, "param2": "foobar"}, parameters)

    def test_positional_manual_placeholder_is_flagged_not_silently_swallowed(self):
        q = (
            Query.from_(self.table_abc)
            .select("*")
            .where((self.table_abc.id == Parameter("?")) & (self.table_abc.name == "foobar"))
        )

        with self.assertRaises(ParameterException):
            q.get_parameterized_sql()

        sql, parameters = q.get_parameterized_sql(strict=False)
        self.assertEqual('SELECT * FROM "abc" WHERE "id"=? AND "name"=?', sql)
        self.assertEqual(["foobar"], parameters)

    def test_subquery_collects_parameters_in_render_order(self):
        subquery = (
            Query.from_(self.table_efg)
            .select(self.table_efg.abc_id)
            .where(self.table_efg.date >= date(2024, 2, 22))
        )
        q = (
            Query.from_(self.table_abc)
            .select("*")
            .where(self.table_abc.category == "foobar")
            .where(self.table_abc.id.isin(subquery))
        )

        sql, parameters = q.get_parameterized_sql()

        self.assertEqual(
            'SELECT * FROM "abc" WHERE "category"=? AND "id" IN (SELECT "abc_id" FROM "efg" WHERE "date">=?)',
            sql,
        )
        self.assertEqual(["foobar", "2024-02-22"], parameters)

    def test_cte_collects_parameters_across_clauses(self):
        cte = Query.from_(self.table_efg).select(self.table_efg.id).where(self.table_efg.total > 100)
        q = (
            Query.with_(cte, "big_orders")
            .from_(self.table_abc)
            .select(self.table_abc.id)
            .where(self.table_abc.id == 42)
        )

        sql, parameters = q.get_parameterized_sql()

        self.assertEqual(
            'WITH big_orders AS (SELECT "id" FROM "efg" WHERE "total">?) SELECT "id" FROM "abc" WHERE "id"=?',
            sql,
        )
        self.assertEqual([100, 42], parameters)

    def test_batch_values_collects_every_value_row_wise(self):
        q = (
            Query.into(self.table_abc)
            .columns("id", "name")
            .insert((1, "a"), (2, "b"), (3, "c"))
        )

        sql, parameters = q.get_parameterized_sql()

        self.assertEqual(
            'INSERT INTO "abc" ("id","name") VALUES (?,?),(?,?),(?,?)',
            sql,
        )
        self.assertEqual([1, "a", 2, "b", 3, "c"], parameters)

    def test_dialect_builders_each_get_fresh_collectors(self):
        cases = {
            MySQLQuery: ("SELECT `id` FROM `abc` WHERE `name`=?", ["foobar"]),
            PostgreSQLQuery: ('SELECT "id" FROM "abc" WHERE "name"=?', ["foobar"]),
            SQLLiteQuery: ('SELECT "id" FROM "abc" WHERE "name"=?', ["foobar"]),
            ClickHouseQuery: ('SELECT "id" FROM "abc" WHERE "name"=?', ["foobar"]),
            MSSQLQuery: ('SELECT "id" FROM "abc" WHERE "name"=?', ["foobar"]),
        }
        for dialect, expected in cases.items():
            with self.subTest(dialect=dialect.__name__):
                query = dialect.from_(self.table_abc).select(self.table_abc.id).where(self.table_abc.name == "foobar")
                self.assertEqual(expected, query.get_parameterized_sql())

    def test_legacy_get_sql_is_unchanged(self):
        q = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")
        parameter = QmarkParameter()

        sql = q.get_sql(parameter=parameter)

        self.assertEqual('SELECT * FROM "abc" WHERE "category"=?', sql)
        self.assertEqual(["foobar"], parameter.get_parameters())
        # And the new entry point still works on the same query afterwards.
        self.assertEqual((sql, ["foobar"]), q.get_parameterized_sql())

    def test_plain_parameter_collector_is_rejected(self):
        q = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")

        with self.assertRaises(ParameterException):
            q.get_parameterized_sql(Parameter(":1"))


class ParameterizedCompileSqliteTests(unittest.TestCase):
    def test_select_with_qmark_executes_in_sqlite(self):
        table = Table("abc")
        q = SQLLiteQuery.from_(table).select(table.id, table.name).where((table.active == 1) & (table.name == "baz"))

        sql, parameters = q.get_parameterized_sql()

        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE abc (id INTEGER, name TEXT, active INTEGER)")
            connection.executemany(
                "INSERT INTO abc VALUES (?,?,?)",
                [(1, "baz", 1), (2, "qux", 0), (3, "baz", 1)],
            )
            rows = connection.execute(sql, parameters).fetchall()

        self.assertEqual([(1, "baz"), (3, "baz")], rows)

    def test_batch_insert_with_qmark_executes_in_sqlite(self):
        table = Table("abc")
        q = SQLLiteQuery.into(table).columns("id", "name").insert((1, "a"), (2, "b"))

        sql, parameters = q.get_parameterized_sql()

        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE abc (id INTEGER, name TEXT)")
            connection.execute(sql, parameters)
            rows = connection.execute('SELECT id, name FROM abc ORDER BY id').fetchall()

        self.assertEqual([(1, "a"), (2, "b")], rows)

    def test_named_parameter_executes_in_sqlite(self):
        table = Table("abc")
        q = SQLLiteQuery.from_(table).select("*").where(table.name == "baz")

        sql, parameters = q.get_parameterized_sql(NamedParameter)

        self.assertEqual('SELECT * FROM "abc" WHERE "name"=:param1', sql)
        self.assertEqual({"param1": "baz"}, parameters)
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE abc (id INTEGER, name TEXT)")
            connection.execute("INSERT INTO abc VALUES (?,?)", (1, "baz"))
            # sqlite3 accepts named placeholders with a matching dict.
            rows = connection.execute(sql, parameters).fetchall()
            self.assertEqual([(1, "baz")], rows)


if __name__ == "__main__":
    unittest.main()
