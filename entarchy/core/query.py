"""Parsing of string filter expressions into the AST dictionaries consumed by the backends.

Grammar (from lowest to highest precedence, i.e. NOT binds tightest):

    or_expr    := xor_expr (OR xor_expr)*
    xor_expr   := and_expr (XOR and_expr)*
    and_expr   := not_expr (AND not_expr)*
    not_expr   := NOT not_expr | primary
    primary    := '(' or_expr ')'
               | EXIST '(' identifier ')'   (parentheses optional)
               | identifier cmp_op operand
               | identifier IN '(' operand (',' operand)* ')'

Comparison operators: ==, !=, <, <=, >, >=
Keywords are case-insensitive: AND, OR, XOR, NOT, IN, EXIST, True, False.
`&`, `|` and `^` are aliases for AND, OR and XOR.

Literal values: 'single'/"double" quoted strings, integers, floats,
ISO dates (2024-01-31) and datetimes (2024-01-31T12:30:00), True/False.
A bare (unquoted) word on the right-hand side of a comparison is treated
as a string value for convenience, e.g. `strain == wildtype`.

Identifiers may contain `/` for grouped attribute names (e.g. s2p/attrs/x),
a leading `../` per parent level, or a `[ParentEntityTypeName]` prefix for
explicit parent traversal.

When the collection is a collection of links, an identifier may also address one
of the link's endpoints with an `@` prefix:

    @Roi.has_receptive_field == True      the endpoint that is a Roi
    @linker.index == 3                    the linker end, by role
    @linked.quality == "good"              the linked end
    @either.strain == "wt"                 at least one endpoint
    @both.has_receptive_field == True      both endpoints
    @linker.[Recording]imaging_rate > 8    an ancestor of the linker

`@linker` and `@linked` are refused for a symmetric kind, where which end is
which is an artifact of uuid ordering rather than meaning.

The resulting AST uses plain dictionaries:
    binary ops: {'left_operand': ..., 'operator': 'AND'|'OR'|'XOR'|cmp|'IN', 'right_operand': ...}
    unary ops:  {'operator': 'NOT'|'EXIST', 'right_operand': ...}
"""
from __future__ import annotations

import datetime
import re
from typing import Any, Union


class QueryParseError(ValueError):
    """Raised when a filter expression cannot be parsed."""


_KEYWORDS = {'AND', 'OR', 'XOR', 'NOT', 'IN', 'EXIST'}

_TOKEN_PATTERN = re.compile(r"""
      (?P<WS>\s+)
    | (?P<STRING_SINGLE>'[^']*')                                        # Strings in single quotes
    | (?P<STRING_DOUBLE>"[^"]*")                                        # Strings in double quotes
    | (?P<DATETIME>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})                 # ISO 8601 datetime
    | (?P<DATE>\d{4}-\d{2}-\d{2})                                       # Date (YYYY-MM-DD)
    | (?P<FLOAT>-?(?:\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+))        # Float numbers (incl. scientific)
    | (?P<INTEGER>-?\d+)                                                # Integer numbers
    | (?P<IDENTIFIER>
          @\w+(?:\.(?:\.\./)*[\w/\[\]]+)?                               # Link endpoint: @Roi.attr, @linker.[Type]attr
        | (?:\.\./|\./)*[\w/\[\]]+                                      # Identifiers (incl. ../ and [Type] prefixes)
      )
    | (?P<OPERATOR>>=|<=|==|!=|<|>)                                     # Comparison operators
    | (?P<AMP>&)
    | (?P<PIPE>\|)
    | (?P<CARET>\^)
    | (?P<COMMA>,)
    | (?P<LPAREN>\()
    | (?P<RPAREN>\))
""", re.VERBOSE)


def tokenize(expression: str) -> list[tuple[str, Any]]:
    """Tokenize an expression into a list of (token_type, value) tuples.

    Raises QueryParseError on characters that are not part of any valid token
    (the previous implementation silently dropped them).
    """

    tokens: list[tuple[str, Any]] = []
    pos = 0
    for match in _TOKEN_PATTERN.finditer(expression):
        if match.start() != pos:
            raise QueryParseError(f'Unrecognized character(s) {expression[pos:match.start()]!r} '
                                  f'at position {pos} in expression {expression!r}')
        pos = match.end()

        kind = match.lastgroup
        value = match.group()

        if kind == 'WS':
            continue
        elif kind in ('STRING_SINGLE', 'STRING_DOUBLE'):
            tokens.append(('VALUE', value[1:-1]))
        elif kind == 'DATETIME':
            tokens.append(('VALUE', datetime.datetime.strptime(value, '%Y-%m-%dT%H:%M:%S')))
        elif kind == 'DATE':
            tokens.append(('VALUE', datetime.datetime.strptime(value, '%Y-%m-%d').date()))
        elif kind == 'FLOAT':
            tokens.append(('VALUE', float(value)))
        elif kind == 'INTEGER':
            tokens.append(('VALUE', int(value)))
        elif kind == 'IDENTIFIER':
            # Bare words may be keywords or boolean literals (case-insensitive)
            _upper = value.upper()
            if _upper in _KEYWORDS:
                tokens.append((_upper, _upper))
            elif _upper in ('TRUE', 'FALSE'):
                tokens.append(('VALUE', _upper == 'TRUE'))
            else:
                tokens.append(('IDENT', value))
        elif kind == 'OPERATOR':
            tokens.append(('CMP', value))
        elif kind == 'AMP':
            tokens.append(('AND', 'AND'))
        elif kind == 'PIPE':
            tokens.append(('OR', 'OR'))
        elif kind == 'CARET':
            tokens.append(('XOR', 'XOR'))
        else:  # COMMA, LPAREN, RPAREN
            tokens.append((kind, value))

    if pos != len(expression):
        raise QueryParseError(f'Unrecognized character(s) {expression[pos:]!r} '
                              f'at position {pos} in expression {expression!r}')

    return tokens


class _Parser:
    """Recursive descent parser with conventional operator precedence
    (comparisons > NOT > AND > XOR > OR).
    """

    def __init__(self, tokens: list[tuple[str, Any]]):
        self._tokens = tokens
        self._index = 0

    def _peek(self) -> tuple[str, Any]:
        if self._index < len(self._tokens):
            return self._tokens[self._index]
        return 'EOF', None

    def _advance(self) -> tuple[str, Any]:
        token = self._peek()
        self._index += 1
        return token

    def _expect(self, kind: str, context: str) -> tuple[str, Any]:
        token = self._advance()
        if token[0] != kind:
            got = 'end of expression' if token[0] == 'EOF' else repr(token[1])
            raise QueryParseError(f'Expected {context}, got {got}')
        return token

    def parse(self) -> dict[str, Any]:
        tree = self._or_expr()

        if self._peek()[0] != 'EOF':
            raise QueryParseError(f'Unexpected input after complete expression, '
                                  f'starting at {self._peek()[1]!r}')

        return tree

    def _or_expr(self) -> dict[str, Any]:
        node = self._xor_expr()
        while self._peek()[0] == 'OR':
            self._advance()
            node = {'left_operand': node, 'operator': 'OR', 'right_operand': self._xor_expr()}
        return node

    def _xor_expr(self) -> dict[str, Any]:
        node = self._and_expr()
        while self._peek()[0] == 'XOR':
            self._advance()
            node = {'left_operand': node, 'operator': 'XOR', 'right_operand': self._and_expr()}
        return node

    def _and_expr(self) -> dict[str, Any]:
        node = self._not_expr()
        while self._peek()[0] == 'AND':
            self._advance()
            node = {'left_operand': node, 'operator': 'AND', 'right_operand': self._not_expr()}
        return node

    def _not_expr(self) -> dict[str, Any]:
        if self._peek()[0] == 'NOT':
            self._advance()
            return {'operator': 'NOT', 'right_operand': self._not_expr()}
        return self._primary()

    def _primary(self) -> dict[str, Any]:
        kind, value = self._peek()

        # Parenthesized sub-expression
        if kind == 'LPAREN':
            self._advance()
            node = self._or_expr()
            self._expect('RPAREN', "')'")
            return node

        # EXIST(attr) or EXIST attr
        if kind == 'EXIST':
            self._advance()
            if self._peek()[0] == 'LPAREN':
                self._advance()
                name = self._expect('IDENT', 'attribute name after EXIST(')[1]
                self._expect('RPAREN', "')' after EXIST(<attribute name>")
            else:
                name = self._expect('IDENT', 'attribute name after EXIST')[1]
            return {'operator': 'EXIST', 'right_operand': name}

        # Comparison: identifier cmp value / identifier IN (...)
        if kind == 'IDENT':
            self._advance()
            op_kind, op_value = self._peek()

            if op_kind == 'CMP':
                self._advance()
                value_kind, value_value = self._advance()
                # Bare words on the right-hand side are treated as string values
                if value_kind not in ('VALUE', 'IDENT'):
                    got = 'end of expression' if value_kind == 'EOF' else repr(value_value)
                    raise QueryParseError(f'Expected a value after "{value} {op_value}", got {got}')
                return {'left_operand': value, 'operator': op_value, 'right_operand': value_value}

            if op_kind == 'IN':
                self._advance()
                return {'left_operand': value, 'operator': 'IN', 'right_operand': self._value_list()}

            got = 'end of expression' if op_kind == 'EOF' else repr(op_value)
            raise QueryParseError(f'Expected a comparison operator or IN after attribute name '
                                  f'{value!r}, got {got}')

        if kind == 'VALUE':
            raise QueryParseError(f'Expressions must start with an attribute name, NOT, EXIST or "(", '
                                  f'got literal value {value!r}')

        got = 'end of expression' if kind == 'EOF' else repr(value)
        raise QueryParseError(f'Expected an expression, got {got}')

    def _value_list(self) -> list[Any]:
        """Parse an IN value list: (v1, v2, ...). Commas are optional for
        backwards compatibility with the previous whitespace-separated syntax.
        """

        self._expect('LPAREN', "'(' after IN")
        values: list[Any] = []

        while True:
            kind, value = self._peek()

            if kind == 'RPAREN':
                self._advance()
                break
            if kind == 'COMMA':
                self._advance()
                continue
            if kind in ('VALUE', 'IDENT'):
                self._advance()
                values.append(value)
                continue

            got = 'end of expression' if kind == 'EOF' else repr(value)
            raise QueryParseError(f'Expected value or \')\' in IN list, got {got}')

        if len(values) == 0:
            raise QueryParseError('IN list must contain at least one value')

        return values


def parse_expression(tokens: list[tuple[str, Any]]) -> Union[dict[str, Any], None]:
    """Parse a token list (as produced by tokenize) into an AST dictionary."""

    if len(tokens) == 0:
        return None

    return _Parser(tokens).parse()


def parse_boolean_expression(expression: str) -> Union[dict[str, Any], None]:
    """Parse a filter expression string into an AST dictionary.

    Returns None for an empty expression. Raises QueryParseError for
    malformed input.
    """

    if expression is None:
        return None

    return parse_expression(tokenize(expression))


def combine_trees(_set_operation: str,
                  left_tree: dict[str, Any],
                  right_tree: Union[dict[str, Any], None] = None):

    _set_operation = _set_operation.upper()

    if _set_operation == 'UNION':
        return {
            'left_operand': left_tree,
            'operator': 'OR',
            'right_operand': right_tree
        }
    elif _set_operation == 'INTERSECTION':
        return {
            'left_operand': left_tree,
            'operator': 'AND',
            'right_operand': right_tree
        }
    elif _set_operation == 'DIFFERENCE':
        return {
            'left_operand': left_tree,
            'operator': 'AND',
            'right_operand': {
                'operator': 'NOT',
                'right_operand': right_tree
            }
        }
    elif _set_operation == 'SYMMETRIC_DIFFERENCE':  # XOR
        return {
            'left_operand': {
                'left_operand': left_tree,
                'operator': 'OR',
                'right_operand': right_tree
            },
            'operator': 'AND',
            'right_operand': {
                'operator': 'NOT',
                'right_operand': {
                    'left_operand': left_tree,
                    'operator': 'AND',
                    'right_operand': right_tree
                }
            }
        }
    elif _set_operation == 'COMPLEMENT':
        return {
            'operator': 'NOT',
            'right_operand': left_tree
        }
    else:
        raise ValueError(f'Unknown set operation: {_set_operation}')


if __name__ == '__main__':

    import pprint

    tree = parse_boolean_expression('attr1 >= 0.55 OR (attr2 == 42 AND NOT(attr3 > 20)) OR EXIST(attr4)')

    pprint.pprint(tree)
