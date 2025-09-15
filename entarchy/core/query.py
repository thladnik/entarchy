from __future__ import annotations
import re
import datetime
import operator



def tokenize(expression):
    # Define a regular expression for matching operands, operators, boolean values, dates, datetimes, and IN/NOT
    token_pattern = r"""
        (?P<STRING_SINGLE>'[^']*')                               # Strings in single quotes
        |(?P<STRING_DOUBLE>"[^"]*")                              # Strings in double quotes
        |(?P<DATETIME>\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b)   # ISO 8601 datetime (YYYY-MM-DDTHH:MM:SS)
        |(?P<DATE>\b\d{4}-\d{2}-\d{2}\b)                         # Date (YYYY-MM-DD)
        |(?P<FLOAT>-?\d+\.\d+)                                   # Float numbers
        |(?P<INTEGER>-?\d+)                                      # Integer numbers
        |(?P<BOOLEAN>\bTrue\b|\bFalse\b)                         # Boolean values
        |(?P<IDENTIFIER>\b[\w/]+\b)                              # Identifiers (like signal1 or name1/subname1)
        |(?P<EXIST>\bEXIST\b)                                    # EXIST operator
        |(?P<IN>\bIN\b)                                          # IN operator
        |(?P<OPERATOR>>=|<=|==|!=|[<>])                          # Comparison operators
        |(?P<LOGICAL>AND|OR|XOR|NOT|&|\|)                        # Logical operators including NOT
        |(?P<LPAREN>\()                                          # Left parenthesis
        |(?P<RPAREN>\))                                          # Right parenthesis
    """

    # Compile the regex
    token_regex = re.compile(token_pattern, re.VERBOSE | re.IGNORECASE)

    # Find all tokens in the expression
    tokens = []
    for match in token_regex.finditer(expression):
        token_type = match.lastgroup
        token_value = match.group(token_type)

        if token_type == 'DATETIME':
            # Convert datetime string to datetime.datetime object
            token_value = datetime.datetime.strptime(token_value, '%Y-%m-%dT%H:%M:%S')
        elif token_type == 'DATE':
            # Convert date string to datetime.date object only if it's not enclosed in quotes
            token_value = datetime.datetime.strptime(token_value, '%Y-%m-%d').date()
        elif token_type == 'FLOAT':
            token_value = float(token_value)  # Convert float string to float
        elif token_type == 'INTEGER':
            token_value = int(token_value)  # Convert integer string to int
        elif token_type == 'BOOLEAN':
            # Keep boolean values as is
            token_value = True if token_value == 'True' else False
        elif token_type == 'STRING_SINGLE' or token_type == 'STRING_DOUBLE':
            # Strip the quotes around the string and treat it as a string
            token_value = token_value[1:-1]

        tokens.append(token_value)

    return tokens


def parse_expression(tokens):
    """
    Recursively parse the tokens into a flattened nested dictionary structure (AST).
    Binary operations have 'left_operand', 'operator', and 'right_operand'.
    Unary operations have 'operator' and 'right_operand'.
    """

    def parse_parentheses(tokens):
        current_expr = None
        while tokens:
            token = tokens.pop(0)

            if token == '(':
                # Start a new group: recursively parse the sub-expression inside parentheses
                sub_expr = parse_parentheses(tokens)
                current_expr = merge_expression(current_expr, sub_expr)
            elif token == ')':
                # Close current group: end of the sub-expression
                break
            elif token.upper() == 'NOT':
                # NOT is a unary operator, it should apply to the next operand or expression
                next_expr = tokens.pop(0)
                if next_expr == '(':
                    parsed_expr = parse_parentheses(tokens)
                else:
                    parsed_expr = next_expr
                current_expr = merge_expression(current_expr, {
                    'operator': 'NOT',
                    'right_operand': parsed_expr
                })
            elif token.upper() in ('AND', 'OR', 'XOR', '&', '|', '^'):
                # Handle AND/OR/XOR operators between expressions
                current_expr = {
                    'left_operand': current_expr,
                    'operator': token.replace('&', 'AND').replace('|', 'OR').replace('^', 'XOR'),
                    'right_operand': parse_parentheses(tokens)
                }
            elif token.upper() == 'EXIST':
                # Handle the EXIST operator, which acts on the next operand
                next_operand = tokens.pop(0)
                if next_operand == '(':
                    next_operand = parse_parentheses(tokens)
                current_expr = merge_expression(current_expr, {
                    'operator': 'EXIST',
                    'right_operand': next_operand
                })
            elif token.upper() == 'IN':
                # Handle the IN operator, which checks if the left operand is in the right list
                left_operand = current_expr
                if tokens[0] == '(':  # Expecting a list enclosed in parentheses
                    tokens.pop(0)  # Remove the '('
                    right_operand = []
                    while tokens[0] != ')':  # Collect all values inside the parentheses
                        right_operand.append(tokens.pop(0))
                    tokens.pop(0)  # Remove the closing ')'
                else:
                    right_operand = tokens.pop(0)
                current_expr = {
                    'left_operand': left_operand,
                    'operator': 'IN',
                    'right_operand': right_operand
                }
            else:
                # Handle comparisons and other binary operations
                if current_expr:
                    left_operand = current_expr
                    operator = token
                    right_operand = tokens.pop(0)
                    current_expr = {
                        'left_operand': left_operand,
                        'operator': operator,
                        'right_operand': right_operand
                    }
                else:
                    current_expr = token

        return current_expr

    def merge_expression(left_expr, right_expr):
        """
        Merges two expressions, ensuring they are combined without unnecessary nesting.
        """
        if left_expr is None:
            return right_expr
        return {
            'left_operand': left_expr,
            'operator': 'AND',  # Default to 'AND' if the operator is implicit
            'right_operand': right_expr
        }

    return parse_parentheses(tokens)


def parse_boolean_expression(expression):
    tokens = tokenize(expression)
    return parse_expression(tokens)



# Object based filters don't work yet, there are many issues...
#
# operator_map = {
#     operator.not_: 'NOT',
#
#     operator.and_: 'AND',
#     operator.or_: 'OR',
#     operator.xor: 'XOR',
#
#     operator.eq: '==',
#     operator.lt: '<',
#     operator.gt: '>',
#     operator.le: '<=',
#     operator.ge: '>=',
#
# }
#
# inv_operator_map = {v: k for k, v in operator_map.items()}
# class Attr:
#
#     def __init__(self, name: str):
#         self.name = name
#
#     def __repr__(self):
#         return f"Attr('{self.name}')"
#
#     def __eq__(self, other: Any):
#         return Filter(self, operator.eq, other)
#
#     def __ne__(self, other: Any):
#         return Filter(self, operator.ne, other)
#
#     def __lt__(self, other: Any):
#         return Filter(self, operator.lt, other)
#
#     def __le__(self, other):
#         return Filter(self, operator.le, other)
#
#     def __gt__(self, other: Any):
#         return Filter(self, operator.gt, other)
#
#     def __ge__(self, other: Any):
#         return Filter(self, operator.ge, other)
#
#     def in_(self, other: Iterable):
#
#         # Make iterable non-mutable
#         if isinstance(other, np.ndarray):
#             other = tuple([n.item() for n in np.array([1, 5, 6])])
#
#         if not isinstance(other, tuple):
#             other = tuple(other)
#
#         return Filter(self, operator.contains, other)
#
#
# class Filter:
#
#     args = None
#
#     def __init__(self, *args):
#
#         # Parse string filter expression
#         if len(args) == 1 and isinstance(args[0], str):
#             parsed_filter = parse_boolean_expression(args[0])
#             args = parsed_filter.args
#
#         # Set args of filter
#         self.args = args
#
#     def __repr__(self):
#         return f"Filter({self.args})"
#
#     def tree(self, level: int = 0):
#
#         _str = ''
#         for arg in self.args:
#             if isinstance(arg, Filter):
#                 _str += arg.tree(level + 1)
#             elif isinstance(arg, Attr):
#                 _str += '\t' * level + str(arg)
#             elif arg in (operator.and_, operator.or_, operator.xor):
#                 _str += '\n' + '\t' * level + f'{operator_map[arg]}\n'
#             elif arg in (operator.eq, operator.lt, operator.gt, operator.le, operator.ge):
#                 _str += ' ' + operator_map[arg] + ' '
#             else:
#                 _str += str(arg) + ' '
#
#         return _str
#
#     # def __rshift__(self, other):
#     #     return Attr(other)
#
#     def __and__(self, other: Union[Filter, str]) -> Filter:
#         if isinstance(other, str):
#             other = Filter(other)
#         return Filter(self, operator.and_, other)
#
#     def __or__(self, other: Filter) -> Filter:
#         return Filter(self, operator.or_, other)
#
#     def __xor__(self, other: Filter) -> Filter:
#         return Filter(self, operator.xor, other)
#
#     def __invert__(self):
#         return Filter(operator.not_, self)
#
#
# def _to_filter(subtree):
#
#     op = subtree['operator']
#     right = subtree['right_operand']
#
#     # Parse next level
#     if isinstance(right, dict):
#         right = _to_filter(right)
#
#     # Unary operation
#     if 'left_operand' not in subtree:
#         return inv_operator_map[op](right)
#
#     left = subtree['left_operand']
#
#     # Parse next level
#     if isinstance(left, dict):
#         left = _to_filter(left)
#     elif isinstance(left, str):
#         left = Attr(left)
#
#     f = inv_operator_map[op](left, right)
#     print(subtree)
#     print('>>', f)
#
#     return f
#
#
# class KeyFilter(object):
#     pass


if __name__ == '__main__':

    import pprint

    tree = parse_boolean_expression('attr1 >= 0.55 OR (attr2 == 42 AND NOT(attr3 > 20)) OR EXIST(attr4)')

    pprint.pprint(tree)

    # f = _to_filter(tree)
