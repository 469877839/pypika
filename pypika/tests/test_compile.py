import sqlite3
import threading
import unittest

from pypika import (
    CompiledQuery,
    FormatParameter,
    NamedParameter,
    NumericParameter,
    Parameter,
    ParameterException,
    ParameterValueWrapper,
    PyformatParameter,
    QmarkParameter,
    Query,
    Table,
    Tables,
    compile_query,
)
from pypika.dialects import (
    ClickHouseQuery,
    MSSQLQuery,
    MySQLQuery,
    PostgreSQLQuery,
    SQLLiteQuery,
)
from pypika.terms import Term


class CompileBasicTests(unittest.TestCase):
    table_abc = Table("abc")

    def test_compile_returns_sql_and_parameters_in_one_call(self):
        query = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")

        compiled = query.compile()

        self.assertIsInstance(compiled, CompiledQuery)
        self.assertEqual('SELECT * FROM "abc" WHERE "category"=?', compiled.sql)
        self.assertEqual(["foobar"], compiled.parameters)

    def test_compile_supports_natural_unpacking(self):
        query = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")

        sql, params = query.compile()

        self.assertEqual('SELECT * FROM "abc" WHERE "category"=?', sql)
        self.assertEqual(["foobar"], params)

    def test_module_level_compile_query(self):
        query = Query.from_(self.table_abc).select(self.table_abc.id).where(self.table_abc.id == 1)

        sql, params = compile_query(query)

        self.assertEqual('SELECT "id" FROM "abc" WHERE "id"=?', sql)
        self.assertEqual([1], params)

    def test_default_style_is_qmark(self):
        query = Query.into(self.table_abc).columns("a", "b").insert(1, "x")

        compiled = query.compile()

        self.assertEqual('INSERT INTO "abc" ("a","b") VALUES (?,?)', compiled.sql)
        self.assertEqual([1, "x"], compiled.parameters)

    def test_compile_accepts_parameter_class_or_instance(self):
        query = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")

        by_class = query.compile(parameter=QmarkParameter)
        by_instance = query.compile(parameter=QmarkParameter())

        self.assertEqual(by_class, by_instance)

    def test_extra_kwargs_are_forwarded_to_get_sql(self):
        query = MySQLQuery.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")

        compiled = query.compile(parameter=FormatParameter)

        self.assertEqual("SELECT * FROM `abc` WHERE `category`=%s", compiled.sql)
        self.assertEqual(["foobar"], compiled.parameters)


class CompileReusabilityTests(unittest.TestCase):
    table_abc = Table("abc")

    def test_repeated_compilation_does_not_accumulate_parameters(self):
        query = (
            Query.from_(self.table_abc)
            .select("*")
            .where(self.table_abc.category == "active")
            .where(self.table_abc.age >= 18)
        )

        first = query.compile()
        second = query.compile()
        third = query.compile(parameter=NamedParameter)

        self.assertEqual(
            'SELECT * FROM "abc" WHERE "category"=? AND "age">=?',
            first.sql,
        )
        self.assertEqual(["active", 18], first.parameters)
        self.assertEqual(first, second)
        self.assertEqual(
            'SELECT * FROM "abc" WHERE "category"=:param1 AND "age">=:param2',
            third.sql,
        )
        self.assertEqual({"param1": "active", "param2": 18}, third.parameters)

    def test_compilation_does_not_mutate_the_parameter_style_template(self):
        template = QmarkParameter()
        query = Query.from_(self.table_abc).select("*").where(self.table_abc.age >= 18)

        query.compile(parameter=template)
        query.compile(parameter=template)

        self.assertEqual([], template.get_parameters())

    def test_compilation_does_not_mutate_the_query_tree(self):
        query = Query.from_(self.table_abc).select(self.table_abc.id).where(self.table_abc.age == 18)
        before = query.get_sql()

        query.compile()
        query.compile(parameter=NamedParameter)

        self.assertEqual(before, query.get_sql())

    def test_concurrent_compilations_are_independent(self):
        query = (
            Query.from_(self.table_abc)
            .select(self.table_abc.id)
            .where(self.table_abc.category == "active")
            .where(self.table_abc.age >= 18)
        )
        expected = (
            'SELECT "id" FROM "abc" WHERE "category"=? AND "age">=?',
            ["active", 18],
        )
        results = []
        errors = []

        def worker():
            try:
                compiled = query.compile()
                results.append((compiled.sql, compiled.parameters))
            except Exception as error:  # pragma: no cover - surfacing thread failures
                errors.append(error)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual([], errors)
        self.assertEqual(50, len(results))
        self.assertTrue(all(result == expected for result in results))

    def test_retry_after_a_failed_compilation_carries_no_partial_state(self):
        class Boom(Term):
            def get_sql(self, **kwargs):
                raise ValueError("boom")

        broken = (
            Query.from_(self.table_abc)
            .select(self.table_abc.id)
            .where(self.table_abc.age == 18)
            .where(Boom() == 1)
        )

        with self.assertRaises(ValueError):
            broken.compile()

        healthy = Query.from_(self.table_abc).select(self.table_abc.id).where(self.table_abc.age == 18)
        compiled = healthy.compile()

        self.assertEqual(['SELECT "id" FROM "abc" WHERE "age"=?', [18]], list(compiled))


class CompilePlaceholderConsistencyTests(unittest.TestCase):
    table_abc, table_efg = Tables("abc", "efg")

    def test_all_positional_styles(self):
        query = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "x").where(
            self.table_abc.age >= 18
        )

        cases = [
            (
                QmarkParameter,
                'SELECT * FROM "abc" WHERE "category"=? AND "age">=?',
                ["x", 18],
            ),
            (
                FormatParameter,
                'SELECT * FROM "abc" WHERE "category"=%s AND "age">=%s',
                ["x", 18],
            ),
            (
                NumericParameter,
                'SELECT * FROM "abc" WHERE "category"=:1 AND "age">=:2',
                ["x", 18],
            ),
        ]
        for style, expected_sql, expected_params in cases:
            with self.subTest(style=style.__name__):
                compiled = query.compile(parameter=style)
                self.assertEqual(expected_sql, compiled.sql)
                self.assertEqual(expected_params, compiled.parameters)

    def test_all_named_styles(self):
        query = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "x")

        named = query.compile(parameter=NamedParameter)
        self.assertEqual('SELECT * FROM "abc" WHERE "category"=:param1', named.sql)
        self.assertEqual({"param1": "x"}, named.parameters)

        pyformat = query.compile(parameter=PyformatParameter)
        self.assertEqual('SELECT * FROM "abc" WHERE "category"=%(param1)s', pyformat.sql)
        self.assertEqual({"param1": "x"}, pyformat.parameters)

    def test_same_term_repeated_in_select_and_where_collects_each_placeholder(self):
        expression = self.table_abc.score + 1
        query = Query.from_(self.table_abc).select(expression.as_("bumped")).where(expression > 10)

        compiled = query.compile()

        self.assertEqual(
            'SELECT "score"+? "bumped" FROM "abc" WHERE "score"+?>?',
            compiled.sql,
        )
        # One bound value per rendered placeholder: the shared term appears twice plus the threshold.
        self.assertEqual([1, 1, 10], compiled.parameters)

    def test_same_named_term_repeated_in_select_and_where(self):
        expression = self.table_abc.score + 1
        query = Query.from_(self.table_abc).select(expression.as_("bumped")).where(expression > 10)

        compiled = query.compile(parameter=NamedParameter)

        self.assertEqual(
            'SELECT "score"+:param1 "bumped" FROM "abc" WHERE "score"+:param2>:param3',
            compiled.sql,
        )
        self.assertEqual({"param1": 1, "param2": 1, "param3": 10}, compiled.parameters)

    def test_subquery_collects_inner_and_outer_parameters_in_order(self):
        subquery = Query.from_(self.table_efg).select(self.table_efg.abc_id).where(self.table_efg.date >= "2024-01-01")
        query = (
            Query.from_(self.table_abc)
            .select("*")
            .where(self.table_abc.category == "foobar")
            .where(self.table_abc.id.isin(subquery))
        )

        compiled = query.compile()

        self.assertEqual(
            'SELECT * FROM "abc" WHERE "category"=? AND "id" IN '
            '(SELECT "abc_id" FROM "efg" WHERE "date">=?)',
            compiled.sql,
        )
        self.assertEqual(["foobar", "2024-01-01"], compiled.parameters)

    def test_cte_collects_parameters_from_with_and_main_query(self):
        cte = Query.from_(self.table_abc).select(self.table_abc.cust_id).where(self.table_abc.total > 50)
        query = (
            Query.with_(cte, "big")
            .from_(Table("big"))
            .select("*")
            .where(Table("big").cust_id == 42)
        )

        compiled = query.compile()

        self.assertEqual(
            'WITH big AS (SELECT "cust_id" FROM "abc" WHERE "total">?) SELECT * FROM "big" WHERE "cust_id"=?',
            compiled.sql,
        )
        self.assertEqual([50, 42], compiled.parameters)

    def test_bulk_values_collects_every_row_in_order(self):
        query = (
            Query.into(self.table_abc)
            .columns("a", "b")
            .insert((1, 1.5), (2, 2.5), (3, None))
        )

        qmark = query.compile()
        self.assertEqual(
            'INSERT INTO "abc" ("a","b") VALUES (?,?),(?,?),(?,null)',
            qmark.sql,
        )
        self.assertEqual([1, 1.5, 2, 2.5, 3], qmark.parameters)

        named = query.compile(parameter=NamedParameter)
        self.assertEqual(
            'INSERT INTO "abc" ("a","b") VALUES (:param1,:param2),(:param3,:param4),(:param5,null)',
            named.sql,
        )
        self.assertEqual(
            {"param1": 1, "param2": 1.5, "param3": 2, "param4": 2.5, "param5": 3},
            named.parameters,
        )

    def test_dialect_builders_compile_with_dialect_specific_rendering(self):
        query = Query.from_(self.table_abc).select(self.table_abc.id).where(self.table_abc.name == "bob")
        cases = [
            (MySQLQuery, 'SELECT `id` FROM `abc` WHERE `name`=?'),
            (PostgreSQLQuery, 'SELECT "id" FROM "abc" WHERE "name"=?'),
            (MSSQLQuery, 'SELECT "id" FROM "abc" WHERE "name"=?'),
            (SQLLiteQuery, 'SELECT "id" FROM "abc" WHERE "name"=?'),
            (ClickHouseQuery, 'SELECT "id" FROM "abc" WHERE "name"=?'),
        ]
        for dialect, expected_sql in cases:
            with self.subTest(dialect=dialect.__name__):
                compiled = dialect.from_(self.table_abc).select(self.table_abc.id).where(
                    self.table_abc.name == "bob"
                ).compile()
                self.assertEqual(expected_sql, compiled.sql)
                self.assertEqual(["bob"], compiled.parameters)

    def test_set_operation_collects_both_branches(self):
        left = Query.from_(self.table_abc).select(self.table_abc.a).where(self.table_abc.x == 1)
        right = Query.from_(self.table_abc).select(self.table_abc.a).where(self.table_abc.y == 2)

        compiled = left.union(right).compile()

        self.assertEqual(
            '(SELECT "a" FROM "abc" WHERE "x"=?) UNION (SELECT "a" FROM "abc" WHERE "y"=?)',
            compiled.sql,
        )
        self.assertEqual([1, 2], compiled.parameters)


class CompileBindingBoundaryTests(unittest.TestCase):
    table_abc = Table("abc")

    def test_bare_parameter_is_rendered_verbatim_and_contributes_no_value(self):
        # A handwritten placeholder with no attached value is an external token: the caller binds it.
        query = (
            Query.from_(self.table_abc)
            .select(self.table_abc.id, Parameter(":tenant"))
            .where(self.table_abc.status == "active")
        )

        compiled = query.compile()

        self.assertEqual('SELECT "id",:tenant FROM "abc" WHERE "status"=?', compiled.sql)
        self.assertEqual(["active"], compiled.parameters)

    def test_handwritten_parameter_value_wrapper_collects_its_bound_value(self):
        query = (
            Query.from_(self.table_abc)
            .select("*")
            .where(self.table_abc.bar == ParameterValueWrapper(NamedParameter("bar"), "bar-value"))
            .where(self.table_abc.status == "active")
        )

        compiled = query.compile(parameter=NamedParameter)

        self.assertEqual(
            'SELECT * FROM "abc" WHERE "bar"=:bar AND "status"=:param2',
            compiled.sql,
        )
        # The handwritten key "bar" occupies a slot, so the auto-generated name starts at param2.
        self.assertEqual({"bar": "bar-value", "param2": "active"}, compiled.parameters)

    def test_mixed_handwritten_and_auto_wrapped_in_positional_style(self):
        query = (
            Query.from_(self.table_abc)
            .select("*")
            .where(self.table_abc.bar == ParameterValueWrapper(Parameter(":bar"), "bar-value"))
            .where(self.table_abc.status == "active")
        )

        compiled = query.compile()

        # The handwritten token keeps its literal spelling; only auto-wrapped values become qmark placeholders.
        self.assertEqual('SELECT * FROM "abc" WHERE "bar"=:bar AND "status"=?', compiled.sql)
        self.assertEqual(["bar-value", "active"], compiled.parameters)

    def test_parameter_value_wrapper_without_collector_raises(self):
        term = ParameterValueWrapper(Parameter(":bar"), "bar-value")

        with self.assertRaises(ParameterException):
            term.get_sql()
        with self.assertRaises(ParameterException):
            Query.from_(self.table_abc).select("*").where(self.table_abc.bar == term).get_sql()

    def test_legacy_get_sql_with_parameter_still_collects(self):
        query = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")
        parameter = QmarkParameter()

        sql = query.get_sql(parameter=parameter)

        self.assertEqual('SELECT * FROM "abc" WHERE "category"=?', sql)
        self.assertEqual(["foobar"], parameter.get_parameters())

    def test_legacy_get_sql_without_parameter_is_unchanged(self):
        query = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")

        self.assertEqual('SELECT * FROM "abc" WHERE "category"=\'foobar\'', query.get_sql())

    def test_stateless_parameter_type_cannot_collect(self):
        query = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")

        with self.assertRaises(ParameterException):
            query.compile(parameter=Parameter("?"))

    def test_stateless_parameter_class_cannot_collect(self):
        query = Query.from_(self.table_abc).select("*").where(self.table_abc.category == "foobar")

        with self.assertRaises(ParameterException):
            query.compile(parameter=Parameter)

    def test_fresh_collector_shares_style_but_not_state(self):
        original = QmarkParameter()
        original.update_parameters("already-there")

        fresh = original.fresh()

        self.assertIsNot(original, fresh)
        self.assertEqual("?", fresh.get_sql())
        self.assertEqual([], fresh.get_parameters())
        self.assertEqual(["already-there"], original.get_parameters())


class CompileSQLiteExecutionTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.executescript(
            """
            CREATE TABLE customers (id INTEGER PRIMARY KEY, tier TEXT);
            CREATE TABLE orders (id INTEGER PRIMARY KEY, cust_id INTEGER, total REAL);
            INSERT INTO customers VALUES (1, 'gold'), (2, 'silver'), (3, 'gold');
            INSERT INTO orders VALUES (10, 1, 150), (11, 2, 80), (12, 3, 200);
            """
        )
        self.customers = Table("customers")
        self.orders = Table("orders")

    def tearDown(self):
        self.connection.close()

    def test_subquery_select_executes_with_qmark_and_named_styles(self):
        query = (
            Query.from_(self.orders)
            .select(self.orders.id, self.orders.total)
            .where(
                self.orders.cust_id.isin(
                    Query.from_(self.customers).select(self.customers.id).where(self.customers.tier == "gold")
                )
            )
            .where(self.orders.total > 100)
        )
        expected = [(10, 150), (12, 200)]

        qmark = query.compile()
        self.assertEqual(expected, self.connection.execute(qmark.sql, qmark.parameters).fetchall())

        named = query.compile(parameter=NamedParameter)
        self.assertEqual(expected, self.connection.execute(named.sql, named.parameters).fetchall())

    def test_cte_executes(self):
        cte = Query.from_(self.orders).select(self.orders.cust_id, self.orders.total).where(self.orders.total >= 100)
        query = Query.with_(cte, "big").from_(Table("big")).select("*").where(Table("big").cust_id == 3)

        compiled = query.compile()

        self.assertEqual([(3, 200)], self.connection.execute(compiled.sql, compiled.parameters).fetchall())

    def test_shared_expression_executes(self):
        expression = self.orders.total * 2
        query = Query.from_(self.orders).select(self.orders.id, expression.as_("doubled")).where(expression > 100)

        compiled = query.compile()

        self.assertEqual(
            [(10, 300), (11, 160), (12, 400)],
            self.connection.execute(compiled.sql, compiled.parameters).fetchall(),
        )

    def test_bulk_values_insert_executes(self):
        self.connection.execute("CREATE TABLE bulk (id INTEGER, total REAL)")
        query = Query.into(Table("bulk")).columns("id", "total").insert((1, 1.5), (2, 2.5), (3, None))

        compiled = query.compile()
        self.connection.execute(compiled.sql, compiled.parameters)

        self.assertEqual(
            [(1, 1.5), (2, 2.5), (3, None)],
            self.connection.execute("SELECT * FROM bulk ORDER BY id").fetchall(),
        )

    def test_mixed_handwritten_and_auto_wrapped_executes_in_named_style(self):
        self.connection.execute("CREATE TABLE mix (a INTEGER, tenant INTEGER)")
        self.connection.execute("INSERT INTO mix VALUES (1, 7), (2, 9)")
        query = (
            Query.from_(Table("mix"))
            .select("a")
            .where(Table("mix").a == ParameterValueWrapper(NamedParameter("need"), 1))
            .where(Table("mix").tenant == 7)
        )

        compiled = query.compile(parameter=NamedParameter)
        self.assertEqual({"need": 1, "param2": 7}, compiled.parameters)

        self.assertEqual([(1,)], self.connection.execute(compiled.sql, compiled.parameters).fetchall())
