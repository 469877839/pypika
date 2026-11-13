import unittest

from pypika import ColumnContract, Query, SetOperationException, Table
from pypika import functions as fn


class OutputContractTests(unittest.TestCase):
    table_a = Table("a")
    table_b = Table("b")
    table_c = Table("c")

    def test_projection_reports_width_names_and_aliases(self):
        query = Query.from_(self.table_a).select(
            self.table_a.id,
            self.table_a.name.as_("label"),
            fn.Sum(self.table_a.amount).as_("total"),
        )

        contract = query.output_columns

        self.assertEqual(3, contract.width)
        self.assertEqual(("id", "label", "total"), contract.names)
        self.assertTrue(contract.names_known)
        self.assertEqual("name", contract[1].source_name)
        self.assertEqual("label", contract[1].alias)

    def test_expression_without_alias_has_known_width_but_unknown_name(self):
        query = Query.from_(self.table_a).select(self.table_a.id, fn.Sum(self.table_a.amount), 1, "fixed")

        contract = query.output_columns

        self.assertEqual(4, contract.width)
        self.assertEqual(("id", None, None, "fixed"), contract.names)
        self.assertFalse(contract.names_known)

    def test_star_and_qualified_star_are_unknown(self):
        for query in (
            Query.from_(self.table_a).select("*"),
            Query.from_(self.table_a).select(self.table_a.star),
            Query.from_(self.table_a).select("a.*"),
        ):
            with self.subTest(sql=str(query)):
                contract = query.output_columns
                self.assertIsInstance(contract, ColumnContract)
                self.assertFalse(contract.known)
                self.assertIsNone(contract.width)
                self.assertIsNone(contract.names)
                self.assertIsNone(contract.columns)

    def test_duplicate_output_names_are_preserved_in_order(self):
        query = Query.from_(self.table_a).select(
            self.table_a.first_name.as_("name"),
            self.table_a.last_name.as_("name"),
        )

        contract = query.output_columns

        self.assertEqual(2, contract.width)
        self.assertEqual(("name", "name"), contract.names)
        self.assertEqual((1, 2), (contract[0].position, contract[1].position))


class SetOperationContractTests(unittest.TestCase):
    table_a = Table("a")
    table_b = Table("b")
    table_c = Table("c")
    table_d = Table("d")

    def test_union_contract_uses_first_branch_output_names(self):
        left = Query.from_(self.table_a).select(
            self.table_a.id,
            self.table_a.name.as_("label"),
        )
        right = Query.from_(self.table_b).select(
            self.table_b.key.as_("other_key"),
            self.table_b.description.as_("other_label"),
        )

        union = left.union(right)

        self.assertEqual(("id", "label"), union.output_columns.names)
        self.assertEqual(
            '(SELECT "id","name" "label" FROM "a") '
            'UNION '
            '(SELECT "key" "other_key","description" "other_label" FROM "b")',
            str(union),
        )

    def test_duplicate_names_survive_nested_set_operation(self):
        left = Query.from_(self.table_a).select(
            self.table_a.id.as_("value"),
            self.table_a.name.as_("value"),
        )
        right = Query.from_(self.table_b).select(
            self.table_b.id.as_("x"),
            self.table_b.name.as_("y"),
        )
        third = Query.from_(self.table_c).select(
            self.table_c.id.as_("p"),
            self.table_c.name.as_("q"),
        )

        query = left.union(right.intersect(third))

        self.assertEqual(2, query.output_columns.width)
        self.assertEqual(("value", "value"), query.output_columns.names)
        self.assertEqual(
            '(SELECT "id" "value","name" "value" FROM "a") '
            'UNION '
            '((SELECT "id" "x","name" "y" FROM "b") '
            'INTERSECT '
            '(SELECT "id" "p","name" "q" FROM "c"))',
            str(query),
        )

    def test_known_column_count_mismatch_has_explicit_diagnostic(self):
        left = Query.from_(self.table_a).select(self.table_a.id, self.table_a.name)
        right = Query.from_(self.table_b).select(self.table_b.id)
        query = left.union(right)

        with self.assertRaises(SetOperationException) as context:
            query.validate_output_columns()

        message = str(context.exception)
        self.assertIn("column count mismatch", message)
        self.assertIn("left operand exposes 2 output column(s)", message)
        self.assertIn("right operand exposes 1 output column(s)", message)
        self.assertIn("id (1), name (2)", message)
        self.assertIn("id (1)", message)

    def test_unknown_width_does_not_guess_or_report_a_mismatch(self):
        left = Query.from_(self.table_a).select("*")
        right = Query.from_(self.table_b).select(self.table_b.id, self.table_b.name)

        query = left.union(right)

        self.assertTrue(query.output_columns.is_unknown)
        self.assertEqual('(SELECT * FROM "a") UNION (SELECT "id","name" FROM "b")', str(query))

    def test_nested_mismatch_is_validated_without_database_execution(self):
        inner = Query.from_(self.table_b).select(self.table_b.id, self.table_b.name).union(
            Query.from_(self.table_c).select(self.table_c.id)
        )
        query = Query.from_(self.table_a).select(self.table_a.id).union(inner)

        with self.assertRaises(SetOperationException):
            query.output_columns

    def test_each_branch_keeps_its_own_orderby_and_limit(self):
        left = (
            Query.from_(self.table_a)
            .select(self.table_a.id)
            .orderby(self.table_a.id)
            .limit(2)
        )
        right = (
            Query.from_(self.table_b)
            .select(self.table_b.id)
            .orderby(self.table_b.id)
            .limit(3)
        )

        query = left.except_of(right)

        self.assertEqual(("id",), query.output_columns.names)
        self.assertEqual(
            '(SELECT "id" FROM "a" ORDER BY "id" LIMIT 2) '
            'EXCEPT '
            '(SELECT "id" FROM "b" ORDER BY "id" LIMIT 3)',
            str(query),
        )


class OuterReferenceTests(unittest.TestCase):
    table_a = Table("a")
    table_b = Table("b")

    def setUp(self):
        self.left = Query.from_(self.table_a).select(
            self.table_a.id.as_("left_id"),
            self.table_a.name.as_("left_name"),
        )
        self.right = Query.from_(self.table_b).select(
            self.table_b.id.as_("right_id"),
            self.table_b.name.as_("right_name"),
        )
        self.union = self.left.union(self.right)

    def test_outer_select_accepts_first_branch_output_names(self):
        query = Query.from_(self.union).select(self.union.left_id, self.union.left_name)

        self.assertEqual(
            'SELECT "sq0"."left_id","sq0"."left_name" '
            'FROM ((SELECT "id" "left_id","name" "left_name" FROM "a") '
            'UNION '
            '(SELECT "id" "right_id","name" "right_name" FROM "b")) "sq0"',
            str(query),
        )

    def test_outer_select_rejects_second_branch_alias(self):
        with self.assertRaises(SetOperationException) as context:
            Query.from_(self.union).select(self.union.right_id)

        message = str(context.exception)
        self.assertIn("references output column 'right_id'", message)
        self.assertIn("left_id (1), left_name (2)", message)
        self.assertIn("first branch", message)

    def test_outer_where_rejects_unknown_set_output(self):
        query = Query.from_(self.union).select(self.union.left_id)

        with self.assertRaises(SetOperationException):
            query.where(self.union.missing == 1)

    def test_outer_reference_allowed_when_output_names_are_unknown(self):
        left = Query.from_(self.table_a).select(fn.Sum(self.table_a.id))
        right = Query.from_(self.table_b).select(fn.Sum(self.table_b.id))
        union = left.union(right)

        query = Query.from_(union).select(union.anything)

        self.assertEqual(
            'SELECT "sq0"."anything" '
            'FROM ((SELECT SUM("id") FROM "a") UNION (SELECT SUM("id") FROM "b")) "sq0"',
            str(query),
        )


class SetOrderByTests(unittest.TestCase):
    table_a = Table("a")
    table_b = Table("b")

    def setUp(self):
        left = Query.from_(self.table_a).select(
            self.table_a.id.as_("left_id"),
            self.table_a.name.as_("left_name"),
        )
        right = Query.from_(self.table_b).select(
            self.table_b.id.as_("right_id"),
            self.table_b.name.as_("right_name"),
        )
        self.union = left.union(right)
        self.right = right

    def test_output_name_and_positional_orderby(self):
        by_name = self.union.orderby("left_name")
        by_position = self.union.orderby(2)

        self.assertTrue(str(by_name).endswith('ORDER BY "left_name"'))
        self.assertTrue(str(by_position).endswith("ORDER BY 2"))

    def test_unknown_output_name_is_rejected(self):
        with self.assertRaises(SetOperationException) as context:
            self.union.orderby("right_name")

        self.assertIn("references output column 'right_name'", str(context.exception))

    def test_branch_alias_is_rejected(self):
        with self.assertRaises(SetOperationException):
            self.union.orderby(self.right.right_name)

    def test_out_of_range_position_is_rejected(self):
        with self.assertRaises(SetOperationException) as context:
            self.union.orderby(3)

        message = str(context.exception)
        self.assertIn("position 3", message)
        self.assertIn("valid positions are 1 through 2", message)

    def test_explicit_validation_catches_legacy_set_field_orderby(self):
        query = self.union.orderby(self.union.field("right_name"))

        with self.assertRaises(SetOperationException):
            query.validate_output_columns()


class PrecedenceTests(unittest.TestCase):
    table_a = Table("a")
    table_b = Table("b")
    table_c = Table("c")

    def test_right_nested_set_operation_keeps_parentheses(self):
        one = Query.from_(self.table_a).select(self.table_a.id)
        two = Query.from_(self.table_b).select(self.table_b.id)
        three = Query.from_(self.table_c).select(self.table_c.id)

        query = one.except_of(two.union(three))

        self.assertEqual(
            '(SELECT "id" FROM "a") '
            'EXCEPT '
            '((SELECT "id" FROM "b") UNION (SELECT "id" FROM "c"))',
            str(query),
        )
        self.assertEqual(("id",), query.output_columns.names)

    def test_operator_precedence_is_not_rewritten(self):
        one = Query.from_(self.table_a).select(self.table_a.id)
        two = Query.from_(self.table_b).select(self.table_b.id)
        three = Query.from_(self.table_c).select(self.table_c.id)

        query = one.union(two.union_all(three))

        self.assertEqual(
            '(SELECT "id" FROM "a") '
            'UNION '
            '((SELECT "id" FROM "b") UNION ALL (SELECT "id" FROM "c"))',
            str(query),
        )


if __name__ == "__main__":
    unittest.main()
