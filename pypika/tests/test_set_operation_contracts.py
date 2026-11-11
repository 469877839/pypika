import sqlite3
import unittest

from pypika import (
    Order,
    OutputColumn,
    OutputColumnContract,
    Query,
    SetOperationException,
    Tables,
    UNKNOWN,
)
from pypika import functions as fn
from pypika.enums import SetOperation


def flat_query():
    """Builder that emits unwrapped compound members, as required by SQLite."""
    return Query._builder(wrap_set_operation_queries=False)


class OutputColumnContractTests(unittest.TestCase):
    table = Tables("abc")[0]

    def test_projection_reports_count_and_names(self):
        query = Query.from_(self.table).select(self.table.foo, self.table.bar.as_("b"))

        contract = query.output_columns

        self.assertTrue(contract.known)
        self.assertEqual(("foo", "b"), contract.names)
        self.assertEqual(2, len(contract))

    def test_unaliased_expression_uses_rendered_text(self):
        query = Query.from_(self.table).select(fn.Count(self.table.foo), self.table.foo + 1, 1)

        self.assertEqual(('COUNT("foo")', '"foo"+1', "1"), query.output_columns.names)
        self.assertEqual(3, len(query.output_columns))

    def test_alias_wins_over_field_name(self):
        query = Query.from_(self.table).select(self.table.foo.as_("renamed"))

        self.assertEqual(("renamed",), query.output_columns.names)

    def test_duplicate_names_keep_positions(self):
        query = Query.from_(self.table).select(self.table.foo.as_("a"), self.table.bar.as_("a"))

        contract = query.output_columns

        self.assertEqual(("a", "a"), contract.names)
        self.assertEqual([0, 1], contract.find_name("a"))
        self.assertEqual(("a",), contract.duplicates())

    def test_star_projection_is_unknown(self):
        for query in (
            Query.from_(self.table).select("*"),
            Query.from_(self.table).select(self.table.star),
        ):
            with self.subTest(query=str(query)):
                contract = query.output_columns

                self.assertFalse(contract.known)
                self.assertIs(UNKNOWN, contract.names)
                self.assertEqual(0, len(contract))
                self.assertIn("*", contract.unknown_reason)

    def test_empty_projection_is_unknown_not_zero_columns(self):
        contract = Query.from_(self.table).output_columns

        self.assertFalse(contract.known)
        self.assertIs(UNKNOWN, contract.names)

    def test_contract_is_inspectable_iterable_sequence(self):
        query = Query.from_(self.table).select(self.table.foo, self.table.bar)
        contract = query.output_columns

        self.assertIsInstance(contract, OutputColumnContract)
        self.assertEqual(["foo", "bar"], [column.name for column in contract])
        self.assertIsInstance(contract[0], OutputColumn)
        self.assertEqual("foo", contract[0].name)
        self.assertEqual("OutputColumnContract(names=('foo', 'bar'))", repr(contract))


class SetOperationCompositionTests(unittest.TestCase):
    table1, table2, table3 = Tables("abc", "efg", "hij")

    def setUp(self):
        self.q1_two = Query.from_(self.table1).select(self.table1.foo, self.table1.bar)
        self.q2_two = Query.from_(self.table2).select(self.table2.foo, self.table2.bar)
        self.q2_three = Query.from_(self.table2).select(self.table2.foo, self.table2.bar, self.table2.baz)

    def test_mismatched_column_counts_raise_for_every_operation(self):
        for operation in ("union", "union_all", "intersect", "except_of", "minus"):
            with self.subTest(operation=operation):
                with self.assertRaises(SetOperationException) as context:
                    getattr(self.q1_two, operation)(self.q2_three)

                self.assertEqual(
                    "Queries combined with {operation} must expose the same number of output columns:"
                    " 2 column(s) ('foo', 'bar') cannot be combined with 3 column(s)"
                    " ('foo', 'bar', 'baz').".format(operation=SetOperation[operation].value),
                    str(context.exception),
                )

    def test_mismatch_raises_when_composing_not_only_when_rendering(self):
        with self.assertRaises(SetOperationException):
            self.q1_two.union(self.q2_three)

        # A valid pair still renders exactly as before.
        self.assertEqual(
            '(SELECT "foo","bar" FROM "abc") UNION (SELECT "foo","bar" FROM "efg")',
            str(self.q1_two.union(self.q2_two)),
        )

    def test_operator_overloads_enforce_the_contract(self):
        for operation in (lambda a, b: a + b, lambda a, b: a * b, lambda a, b: a - b):
            with self.subTest(operation=operation):
                with self.assertRaises(SetOperationException):
                    operation(self.q1_two, self.q2_three)

    def test_chained_union_validates_each_appended_branch(self):
        chain = self.q1_two.union(self.q2_two)

        with self.assertRaises(SetOperationException):
            chain.union(self.q2_three)

        self.assertEqual(1, len(chain._set_operation))
        self.assertEqual(3, len(chain.union(self.q2_two).union(self.q2_two)._set_operation))

    def test_three_branch_chain_contract_uses_first_branch(self):
        q3_two = Query.from_(self.table3).select(self.table3.alias_a, self.table3.alias_b)
        chain = self.q1_two.union(self.q2_two).intersect(q3_two)

        self.assertEqual(("foo", "bar"), chain.output_columns.names)

    def test_contract_names_come_from_leftmost_branch_not_right_branch_aliases(self):
        left = Query.from_(self.table1).select(self.table1.foo.as_("key"), self.table1.bar)
        right = Query.from_(self.table2).select(self.table2.foo.as_("ignored"), self.table2.bar.as_("ignored2"))

        unioned = left.union(right)

        self.assertEqual(("key", "bar"), unioned.output_columns.names)

    def test_nested_set_operation_contracts(self):
        inner = Query.from_(self.table2).select(self.table2.foo, self.table2.bar).union(
            Query.from_(self.table3).select(self.table3.foo, self.table3.bar)
        )
        outer = Query.from_(self.table1).select(self.table1.foo.as_("x"), self.table1.bar).union(inner)

        self.assertEqual(("x", "bar"), outer.output_columns.names)

        with self.assertRaises(SetOperationException) as context:
            self.q1_two.union(inner.union(Query.from_(self.table3).select(self.table3.one)))
        self.assertEqual(
            "Queries combined with UNION must expose the same number of output columns:"
            " 2 column(s) ('foo', 'bar') cannot be combined with 1 column(s) ('one',).",
            str(context.exception),
        )

    def test_unknown_branch_defers_validation_to_the_database(self):
        star_branch = Query.from_(self.table1).select("*")
        explicit_branch = Query.from_(self.table2).select(self.table2.foo, self.table2.bar, self.table2.baz)

        unioned = star_branch.union(explicit_branch)

        contract = unioned.output_columns
        self.assertFalse(contract.known)
        self.assertIs(UNKNOWN, contract.names)
        # Composition itself is still allowed; only the contract stays unknown.
        self.assertIn("UNION", str(unioned))

    def test_render_rechecks_the_contract_if_internals_were_mutated(self):
        unioned = self.q1_two.union(self.q2_two)
        unioned._set_operation.append(
            (SetOperation.union, Query.from_(self.table3).select(self.table3.foo, self.table3.bar, self.table3.baz))
        )

        with self.assertRaises(SetOperationException) as context:
            unioned.get_sql()

        self.assertIn("must expose the same number of output columns", str(context.exception))


class OuterOrderByContractTests(unittest.TestCase):
    table1, table2 = Tables("abc", "efg")

    def setUp(self):
        self.left = Query.from_(self.table1).select(self.table1.foo.as_("a"), self.table1.bar)
        self.right = Query.from_(self.table2).select(self.table2.baz, self.table2.buz)
        self.unioned = self.left.union(self.right)

    def test_orderby_output_name_is_allowed(self):
        query = self.unioned.orderby("a").orderby("bar")

        self.assertIn('ORDER BY "a","bar"', str(query))

    def test_orderby_position_is_allowed_within_range(self):
        self.assertEqual(
            '(SELECT "foo" "a","bar" FROM "abc") UNION (SELECT "baz","buz" FROM "efg") ORDER BY 2',
            str(self.unioned.orderby(2)),
        )

    def test_orderby_position_outside_range_is_rejected(self):
        for position in (0, 3, 9, -1):
            with self.subTest(position=position):
                with self.assertRaises(SetOperationException) as context:
                    self.unioned.orderby(position)

                self.assertEqual(
                    "ORDER BY position {position} is out of range for a set operation with 2 output column(s)"
                    " ('a', 'bar').".format(position=position),
                    str(context.exception),
                )

    def test_orderby_name_missing_from_output_contract_is_rejected(self):
        with self.assertRaises(SetOperationException) as context:
            self.unioned.orderby("buz")

        self.assertEqual(
            "ORDER BY name 'buz' does not match any output column of the set operation ('a', 'bar');"
            " outer ORDER BY cannot reference branch-only columns.",
            str(context.exception),
        )

    def test_branch_only_alias_is_rejected_even_when_it_exists_on_the_right_branch(self):
        left = Query.from_(self.table1).select(self.table1.foo.as_("a"))
        right = Query.from_(self.table2).select(self.table2.bar.as_("z"))

        with self.assertRaises(SetOperationException) as context:
            left.union(right).orderby("z")

        self.assertIn("does not match any output column", str(context.exception))
        self.assertEqual(("a",), left.union(right).output_columns.names)

    def test_branch_table_field_resolved_against_output_names(self):
        self.assertIn('ORDER BY "bar"', str(self.unioned.orderby(self.table1.bar)))

        with self.assertRaises(SetOperationException):
            self.unioned.orderby(self.table2.buz)

    def test_union_field_must_match_output_contract(self):
        self.assertIn('ORDER BY "a"', str(self.unioned.orderby(self.unioned.field("a"))))

        with self.assertRaises(SetOperationException):
            self.unioned.orderby(self.unioned.field("nope"))

    def test_expression_over_output_name_is_allowed(self):
        query = self.unioned.orderby(fn.Lower(self.unioned.field("a")))

        self.assertIn('ORDER BY LOWER("a")', str(query))

    def test_unknown_contract_skips_orderby_validation(self):
        star_union = Query.from_(self.table1).select("*").union(Query.from_(self.table2).select("*"))

        query = star_union.orderby("anything").orderby(99)

        self.assertIn('ORDER BY "anything",99', str(query))

    def test_duplicate_output_names_keep_positions_unambiguous(self):
        duplicated = Query.from_(self.table1).select(
            self.table1.foo.as_("a"), self.table1.bar.as_("a")
        ).union(Query.from_(self.table2).select(self.table2.baz, self.table2.buz))

        self.assertEqual(("a", "a"), duplicated.output_columns.names)
        self.assertEqual([0, 1], duplicated.output_columns.find_name("a"))
        self.assertIn('ORDER BY "a",2', str(duplicated.orderby("a").orderby(2)))

    def test_boolean_literal_is_not_treated_as_position(self):
        query = self.unioned.orderby(True)

        self.assertIn("ORDER BY true", str(query))

    def test_orderby_does_not_mutate_original_set_operation(self):
        before = list(self.unioned._orderbys)
        self.unioned.orderby("a")

        self.assertEqual(before, self.unioned._orderbys)


class WrappingQueryReferenceTests(unittest.TestCase):
    table1, table2 = Tables("abc", "efg")

    def setUp(self):
        self.unioned = Query.from_(self.table1).select(self.table1.foo.as_("a"), self.table1.bar).union(
            Query.from_(self.table2).select(self.table2.baz, self.table2.buz)
        )

    def test_valid_output_names_can_be_selected_and_ordered(self):
        query = Query.from_(self.unioned).select(self.unioned.a, self.unioned.bar).orderby("bar")

        self.assertIn('SELECT "sq0"."a","sq0"."bar"', str(query))
        self.assertIn('ORDER BY "sq0"."bar"', str(query))

    def test_selecting_a_missing_output_column_is_rejected(self):
        with self.assertRaises(SetOperationException) as context:
            Query.from_(self.unioned).select(self.unioned.buz)

        self.assertEqual(
            "Reference 'buz' does not match any output column of the set operation ('a', 'bar')."
            " Outer queries can only reference the set operation's output columns.",
            str(context.exception),
        )

    def test_orderby_on_a_missing_output_column_is_rejected(self):
        with self.assertRaises(SetOperationException):
            Query.from_(self.unioned).select(self.unioned.a).orderby("nope")

    def test_unknown_contract_keeps_wrapper_references_permissive(self):
        star_union = Query.from_(self.table1).select("*").union(Query.from_(self.table2).select("*"))

        query = Query.from_(star_union).select(star_union.anything)

        self.assertIn('"sq0"."anything"', str(query))


class BranchClauseTests(unittest.TestCase):
    table1, table2, table3 = Tables("abc", "efg", "hij")

    def setUp(self):
        self.q1 = Query.from_(self.table1).select(self.table1.foo, self.table1.bar)
        self.q2 = Query.from_(self.table2).select(self.table2.foo, self.table2.bar)
        self.q3 = Query.from_(self.table3).select(self.table3.foo, self.table3.bar)

    def test_each_branch_keeps_its_own_orderby_and_limit(self):
        inner = self.q2.union(self.q3).orderby(self.table2.foo).limit(2).offset(1)
        outer = self.q1.union(inner).orderby(self.table1.foo).limit(5).offset(3)

        self.assertEqual(
            '(SELECT "foo","bar" FROM "abc") UNION ('
            '(SELECT "foo","bar" FROM "efg") UNION (SELECT "foo","bar" FROM "hij")'
            ' ORDER BY "foo" LIMIT 2 OFFSET 1) ORDER BY "foo" LIMIT 5 OFFSET 3',
            str(outer),
        )
        self.assertEqual(("foo", "bar"), outer.output_columns.names)

    def test_branch_clauses_do_not_change_the_contract(self):
        inner = self.q2.orderby(self.table2.foo).limit(10).union(
            self.q3.orderby(self.table3.bar, order=Order.desc)
        )

        self.assertEqual(("foo", "bar"), inner.output_columns.names)


class PrecedenceRegressionTests(unittest.TestCase):
    table1, table2, table3, table4 = Tables("abc", "efg", "hij", "lmn")

    def setUp(self):
        self.q1 = Query.from_(self.table1).select("foo")
        self.q2 = Query.from_(self.table2).select("foo")
        self.q3 = Query.from_(self.table3).select("foo")
        self.q4 = Query.from_(self.table4).select("foo")

    def test_nested_right_operand_keeps_parentheses(self):
        self.assertEqual(
            '(SELECT "foo" FROM "abc") UNION ((SELECT "foo" FROM "efg") UNION (SELECT "foo" FROM "hij"))',
            str(self.q1.union(self.q2.union(self.q3))),
        )

    def test_flat_chain_has_no_extra_parentheses(self):
        self.assertEqual(
            '(SELECT "foo" FROM "abc") UNION (SELECT "foo" FROM "efg") INTERSECT (SELECT "foo" FROM "hij")',
            str(self.q1.union(self.q2).intersect(self.q3)),
        )

    def test_deeply_nested_operand_preserves_grouping(self):
        self.assertEqual(
            '(SELECT "foo" FROM "abc") UNION ((SELECT "foo" FROM "efg") UNION '
            '((SELECT "foo" FROM "hij") INTERSECT (SELECT "foo" FROM "lmn")))',
            str(self.q1.union(self.q2.union(self.q3.intersect(self.q4)))),
        )

    def test_arithmetic_precedence_is_unchanged(self):
        table = Tables("xyz")[0]
        self.assertEqual(
            'SELECT ("a"+"b")*"c" FROM "xyz"',
            str(Query.from_(table).select((table.a + table.b) * table.c)),
        )
        self.assertEqual(
            'SELECT "a"+"b"*"c" FROM "xyz"',
            str(Query.from_(table).select(table.a + table.b * table.c)),
        )
        self.assertEqual(
            'SELECT "a"-("b"-"c") FROM "xyz"',
            str(Query.from_(table).select(table.a - (table.b - table.c))),
        )


class _GuardedConnection:
    """Wraps a connection and fails loudly if anything tries to execute SQL."""

    def __init__(self):
        self.calls = 0

    def execute(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("validation must not execute a database query")


class ExecutedSetOperationExampleTests(unittest.TestCase):
    """
    Examples that are actually executed against an in-memory SQLite database.

    They prove three properties with real engine feedback: the column contract matches the database's
    reported output columns, output names come from the left-most branch, and parentheses change the
    meaning of a nested set operation.
    """

    @classmethod
    def setUpClass(cls):
        cls.db = sqlite3.connect(":memory:")
        cls.db.executescript(
            """
            CREATE TABLE t1 (id INTEGER, name TEXT, score INTEGER);
            CREATE TABLE t2 (id INTEGER, label TEXT, points INTEGER);
            CREATE TABLE t3 (id INTEGER, name TEXT, score INTEGER);
            INSERT INTO t1 VALUES (1,'a',10),(2,'b',20);
            INSERT INTO t2 VALUES (1,'a',10),(3,'c',30);
            INSERT INTO t3 VALUES (2,'b',20),(4,'d',40);
            """
        )
        cls.t1, cls.t2, cls.t3 = Tables("t1", "t2", "t3")

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def _execute(self, query):
        cursor = self.db.execute(query.get_sql(quote_char=None))
        return cursor.fetchall(), tuple(description[0] for description in cursor.description)

    def test_contract_names_match_database_description(self):
        left = flat_query().from_(self.t1).select(self.t1.id.as_("key"), self.t1.name.as_("label"))
        right = flat_query().from_(self.t2).select(self.t2.id, self.t2.label)
        unioned = left.union(right)

        rows, described = self._execute(unioned)

        self.assertEqual(unioned.output_columns.names, described)
        self.assertEqual([(1, "a"), (2, "b"), (3, "c")], rows)

    def test_output_names_come_from_leftmost_branch(self):
        left = flat_query().from_(self.t1).select(self.t1.id.as_("key"))
        right = flat_query().from_(self.t2).select(self.t2.id.as_("branch_two_only"))
        unioned = left.union(right)

        rows, described = self._execute(unioned)

        self.assertEqual(("key",), described)
        self.assertEqual(("key",), unioned.output_columns.names)
        self.assertEqual([(1,), (2,), (3,)], rows)

        # SQLite happens to accept ordering by the right branch alias; the contract rejects it because
        # the portable, standard-compliant output name is "key".
        with self.assertRaises(SetOperationException):
            unioned.orderby("branch_two_only")

    def test_star_contract_stays_unknown_while_database_reports_real_columns(self):
        unioned = flat_query().from_(self.t1).select("*").union(flat_query().from_(self.t2).select("*"))

        rows, described = self._execute(unioned)

        self.assertFalse(unioned.output_columns.known)
        self.assertIs(UNKNOWN, unioned.output_columns.names)
        # The database knows the real expansion; PyPika deliberately does not guess it.
        self.assertEqual(("id", "name", "score"), described)
        self.assertEqual([(1, "a", 10), (2, "b", 20), (3, "c", 30)], rows)

    def test_positional_and_named_outer_orderby_execute(self):
        left = flat_query().from_(self.t1).select(self.t1.id.as_("key"), self.t1.name.as_("label"))
        right = flat_query().from_(self.t2).select(self.t2.id, self.t2.label)
        unioned = left.union(right)

        rows_by_name, _ = self._execute(unioned.orderby("key", order=Order.desc))
        rows_by_position, _ = self._execute(unioned.orderby(1, order=Order.desc))

        self.assertEqual([(3, "c"), (2, "b"), (1, "a")], rows_by_name)
        self.assertEqual(rows_by_name, rows_by_position)

    def test_validation_rejects_misuse_without_touching_the_database(self):
        guard = _GuardedConnection()
        left = flat_query().from_(self.t1).select(self.t1.id.as_("key"))
        right = flat_query().from_(self.t2).select(self.t2.id)

        unioned = left.union(right)
        with self.assertRaises(SetOperationException):
            unioned.orderby(2)
        with self.assertRaises(SetOperationException):
            unioned.orderby("label")
        with self.assertRaises(SetOperationException):
            left.union(flat_query().from_(self.t2).select(self.t2.id, self.t2.label))

        self.assertEqual(0, guard.calls)

    def test_parentheses_change_the_meaning_of_a_set_operation(self):
        a = flat_query().from_(self.t1).select(self.t1.id)
        b = flat_query().from_(self.t2).select(self.t2.id)
        c = flat_query().from_(self.t3).select(self.t3.id)

        # (t1 UNION t2) INTERSECT t3 -> intersection applies to the combined result.
        left_grouped = a.union(b).intersect(c)
        rows_left, _ = self._execute(left_grouped)
        self.assertEqual([(2,)], rows_left)
        self.assertEqual(("id",), left_grouped.output_columns.names)

        # t1 UNION (t2 INTERSECT t3) -> intersection is applied before the union.
        right_operand = flat_query().from_(b.intersect(c)).select("id")
        right_grouped = a.union(right_operand)
        rows_right, _ = self._execute(right_grouped)
        self.assertEqual([(1,), (2,)], rows_right)
        self.assertEqual(("id",), right_grouped.output_columns.names)

        self.assertNotEqual(rows_left, rows_right)

    def test_each_branch_keeps_its_own_orderby_and_limit(self):
        inner = (
            flat_query()
            .from_(self.t2)
            .select(self.t2.id, self.t2.label)
            .union(flat_query().from_(self.t3).select(self.t3.id, self.t3.name))
            .orderby(self.t2.id)
            .limit(2)
        )
        outer = Query.from_(inner).select("*")

        rows, described = self._execute(outer)

        self.assertEqual([(1, "a"), (2, "b")], rows)
        self.assertEqual(inner.output_columns.names, described)
        # The wrapping query projects "*", so its own contract is unknown even though the nested set
        # operation has a known one.
        self.assertEqual(("id", "label"), inner.output_columns.names)
        self.assertFalse(outer.output_columns.known)


if __name__ == "__main__":
    unittest.main()
