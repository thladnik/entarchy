"""Console capability helpers.

The default alive_progress styles use Unicode characters that legacy Windows
consoles (cp1252 etc.) cannot encode, which crashes the progress bar with a
UnicodeEncodeError. These helpers pick equivalent ASCII styles on such consoles.
"""
import sys


def stdout_supports_unicode() -> bool:
    encoding = getattr(sys.stdout, 'encoding', None) or ''
    return encoding.replace('-', '').lower().startswith('utf')


def bar_style(**overrides) -> dict:
    """Return alive_progress style kwargs that are safe for the current console.

    Keyword arguments override the defaults, e.g. bar_style(bar=None, length=20).
    """
    if stdout_supports_unicode():
        style = {'spinner': 'fish2', 'spinner_length': 30}
    else:
        style = {'spinner': 'classic', 'spinner_length': 12, 'bar': 'classic'}

    style.update(overrides)
    return style
