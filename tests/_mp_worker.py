"""Worker functions for map_async tests.

Kept in an importable module (not the test file), because spawned worker
processes must be able to import the function by qualified name.
"""


def double_score(session):
    session['doubled'] = float(session['score']) * 2.0
