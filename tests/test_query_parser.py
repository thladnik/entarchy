import datetime

import pytest

from entarchy.core import query
from entarchy.core.query import QueryParseError, parse_boolean_expression


def cmp(name, op, value):
    return {'left_operand': name, 'operator': op, 'right_operand': value}


class TestPrecedence:

    def test_and_binds_tighter_than_or(self):
        tree = parse_boolean_expression('a == 1 AND b == 2 OR c == 3')
        assert tree == {
            'left_operand': {'left_operand': cmp('a', '==', 1), 'operator': 'AND',
                             'right_operand': cmp('b', '==', 2)},
            'operator': 'OR',
            'right_operand': cmp('c', '==', 3),
        }

    def test_or_then_and(self):
        tree = parse_boolean_expression('a == 1 OR b == 2 AND c == 3')
        assert tree == {
            'left_operand': cmp('a', '==', 1),
            'operator': 'OR',
            'right_operand': {'left_operand': cmp('b', '==', 2), 'operator': 'AND',
                              'right_operand': cmp('c', '==', 3)},
        }

    def test_parentheses_override(self):
        tree = parse_boolean_expression('a == 1 AND (b == 2 OR c == 3)')
        assert tree == {
            'left_operand': cmp('a', '==', 1),
            'operator': 'AND',
            'right_operand': {'left_operand': cmp('b', '==', 2), 'operator': 'OR',
                              'right_operand': cmp('c', '==', 3)},
        }

    def test_left_associativity(self):
        tree = parse_boolean_expression('a == 1 AND b == 2 AND c == 3')
        assert tree == {
            'left_operand': {'left_operand': cmp('a', '==', 1), 'operator': 'AND',
                             'right_operand': cmp('b', '==', 2)},
            'operator': 'AND',
            'right_operand': cmp('c', '==', 3),
        }

    def test_xor_between_and_and_or(self):
        tree = parse_boolean_expression('a == 1 OR b == 2 XOR c == 3 AND d == 4')
        # AND binds tighter than XOR, XOR tighter than OR
        assert tree['operator'] == 'OR'
        assert tree['right_operand']['operator'] == 'XOR'
        assert tree['right_operand']['right_operand']['operator'] == 'AND'

    def test_symbol_aliases(self):
        assert (parse_boolean_expression('a == 1 & b == 2 | c == 3')
                == parse_boolean_expression('a == 1 AND b == 2 OR c == 3'))
        assert parse_boolean_expression('a == 1 ^ b == 2')['operator'] == 'XOR'


class TestNot:

    def test_not_binds_to_comparison(self):
        tree = parse_boolean_expression('NOT a == 1 AND b == 2')
        assert tree == {
            'left_operand': {'operator': 'NOT', 'right_operand': cmp('a', '==', 1)},
            'operator': 'AND',
            'right_operand': cmp('b', '==', 2),
        }

    def test_not_with_parens_is_equivalent(self):
        assert (parse_boolean_expression('NOT(a > 20)')
                == parse_boolean_expression('NOT a > 20'))

    def test_double_not(self):
        tree = parse_boolean_expression('NOT NOT a == 1')
        assert tree == {'operator': 'NOT',
                        'right_operand': {'operator': 'NOT', 'right_operand': cmp('a', '==', 1)}}

    def test_not_exist(self):
        tree = parse_boolean_expression('NOT(EXIST(attr))')
        assert tree == {'operator': 'NOT',
                        'right_operand': {'operator': 'EXIST', 'right_operand': 'attr'}}


class TestLiterals:

    @pytest.mark.parametrize('literal,expected', [
        ('True', True), ('true', True), ('TRUE', True),
        ('False', False), ('false', False), ('FALSE', False),
    ])
    def test_boolean_case_insensitive(self, literal, expected):
        tree = parse_boolean_expression(f'flag == {literal}')
        assert tree['right_operand'] is expected

    def test_numbers(self):
        assert parse_boolean_expression('a == -5')['right_operand'] == -5
        assert parse_boolean_expression('a == 1.25')['right_operand'] == 1.25
        assert parse_boolean_expression('a == 1.5e3')['right_operand'] == 1500.0
        assert parse_boolean_expression('a == -2e-2')['right_operand'] == -0.02

    def test_date_and_datetime(self):
        assert parse_boolean_expression('d == 2024-01-31')['right_operand'] == datetime.date(2024, 1, 31)
        assert (parse_boolean_expression('d == 2024-01-31T12:30:00')['right_operand']
                == datetime.datetime(2024, 1, 31, 12, 30, 0))

    def test_quoted_strings_are_never_keywords(self):
        assert parse_boolean_expression('name == "AND"')['right_operand'] == 'AND'
        assert parse_boolean_expression("name == 'True'")['right_operand'] == 'True'

    def test_bare_word_rhs_is_string(self):
        assert parse_boolean_expression('strain == wildtype')['right_operand'] == 'wildtype'


class TestIdentifiers:

    def test_keyword_substrings_are_identifiers(self):
        for name in ['android', 'income', 'notes', 'organisation', 'exist_flag']:
            tree = parse_boolean_expression(f'{name} == 1')
            assert tree['left_operand'] == name

    def test_parent_traversal_forms(self):
        assert parse_boolean_expression('../strain == "x"')['left_operand'] == '../strain'
        assert parse_boolean_expression('[Subject]strain == "x"')['left_operand'] == '[Subject]strain'
        assert (parse_boolean_expression('display/__visual_name == "Y"')['left_operand']
                == 'display/__visual_name')


class TestOperators:

    def test_not_equal(self):
        assert parse_boolean_expression('a != 3')['operator'] == '!='

    def test_in_with_commas(self):
        tree = parse_boolean_expression('a IN (1, 2, 3)')
        assert tree == cmp('a', 'IN', [1, 2, 3])

    def test_in_without_commas(self):
        assert parse_boolean_expression('a IN (1 2 3)') == cmp('a', 'IN', [1, 2, 3])

    def test_in_with_strings(self):
        tree = parse_boolean_expression('strain IN ("wt", "mut")')
        assert tree == cmp('strain', 'IN', ['wt', 'mut'])

    def test_exist_forms(self):
        assert (parse_boolean_expression('EXIST(attr)')
                == parse_boolean_expression('EXIST attr')
                == {'operator': 'EXIST', 'right_operand': 'attr'})


class TestErrors:

    def test_empty_expression_is_none(self):
        assert parse_boolean_expression('') is None
        assert parse_boolean_expression(None) is None

    @pytest.mark.parametrize('expression', [
        'a ==',                    # missing right operand
        '(a == 1',                 # unbalanced open paren
        'a == 1)',                 # unbalanced close paren
        'a === 1',                 # invalid operator remainder
        'a = 1',                   # invalid operator remainder
        'a == 1 $$$ b == 2',       # garbage characters
        'a == 1 AND AND b == 2',   # doubled connective
        'NOT',                     # dangling NOT
        'a IN',                    # dangling IN
        'a IN ()',                 # empty IN list
        'a IN (1, 2',              # unterminated IN list
        '5 == a',                  # literal on left side
        '5',                       # bare literal
        'a == 1 b == 2',           # missing connective
    ])
    def test_malformed_raises_parse_error(self, expression):
        with pytest.raises(QueryParseError):
            parse_boolean_expression(expression)


class TestCombineTrees:

    def test_union_intersection(self):
        a, b = cmp('a', '==', 1), cmp('b', '==', 2)
        assert query.combine_trees('UNION', a, b)['operator'] == 'OR'
        assert query.combine_trees('INTERSECTION', a, b)['operator'] == 'AND'
        assert query.combine_trees('COMPLEMENT', a)['operator'] == 'NOT'
