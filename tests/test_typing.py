"""The annotations, checked by a type checker rather than at runtime.

`Collection` being generic has no runtime effect whatsoever, so nothing else in
this suite would notice if it stopped carrying its entity type - and the whole
point of it is what an editor says about `ent.get(Roi)[0]` before anything runs.
So this asks pyright, and skips where pyright is not installed.
"""
import json
import pathlib
import subprocess
import sys
import sysconfig

import pytest

pytest.importorskip('pyright', reason='typing is checked by pyright when present')

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Its own tiny schema rather than conftest's, so that a schema's own members -
#  the thing all of this exists for - are here to be asked about
PROBE = '''
import entarchy


class Roi(entarchy.Entity):
    @property
    def peak(self) -> float:
        ...


class Layer(entarchy.Entity):
    pass


Layer.add_child_entity_type(Roi)


class Archy(entarchy.Entarchy):
    _implementation_version = '0.1'
    _hierarchy_root_type = Layer


def probe(ent: Archy) -> None:
    rois = ent.get(Roi)

    reveal_type(rois)
    reveal_type(rois[0])
    reveal_type(rois[0].peak)
    reveal_type(rois[0:2])
    reveal_type(rois['index'])
    reveal_type(rois[['index', 'good']])
    reveal_type(rois.sort('index'))
    reveal_type(rois.sort_by_hierarchy())
    reveal_type(rois.where('index > 1'))
    reveal_type(~rois)
    reveal_type(rois.entity_type)
    reveal_type(rois.get_entity('u', 'i'))
    reveal_type(Roi('index > 1'))
    reveal_type(Roi('index > 1').get_from(ent))
    reveal_type(ent.get('Roi'))
    reveal_type(ent.get('Roi')[0])

    for roi in rois:
        reveal_type(roi)
'''

# In the order they appear above
EXPECTED = [
    'Collection[Roi]',
    'Roi',
    # A collection of Rois gives back Rois, so the schema's own members resolve
    'float',
    'list[Roi]',
    'Series',
    'DataFrame',
    # Everything that hands back another collection keeps the entity type
    'Collection[Roi]',
    'Collection[Roi]',
    'Collection[Roi]',
    'Collection[Roi]',
    'type[Roi]',
    'Roi',
    # The deferred form carries it too
    'DeferredEntityCollection[Roi]',
    'Collection[Roi]',
    # Naming a type as a string cannot say which one, and must not pretend to
    'Collection[Entity]',
    'Entity',
    'Roi',
]


def run_pyright(tmp_path) -> list[str]:
    """The types pyright reports for the probe, in order.

    Skips rather than fails when pyright cannot start: the pip package fetches
    a node binary on first use, which an offline machine will not have.
    """
    (tmp_path / 'probe.py').write_text(PROBE, encoding='utf-8')
    (tmp_path / 'pyrightconfig.json').write_text(json.dumps({
        'include': ['probe.py'],
        # The working tree first, so this checks the source in hand rather than
        #  whatever version happens to be installed; then site-packages, named
        #  outright because the probe runs from a temporary directory with no
        #  interpreter of its own for pyright to work it out from
        'extraPaths': [REPO_ROOT.as_posix(),
                       pathlib.Path(sysconfig.get_paths()['purelib']).as_posix()],
        'typeCheckingMode': 'basic',
    }), encoding='utf-8')

    try:
        completed = subprocess.run(
            [sys.executable, '-m', 'pyright', '--outputjson'],
            cwd=tmp_path, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f'pyright could not be run: {exc}')

    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        pytest.skip(f'pyright produced no report: {completed.stderr[:400]}')

    errors = [d for d in report['generalDiagnostics'] if d['severity'] == 'error']
    assert errors == [], '\n'.join(
        f"probe.py:{d['range']['start']['line'] + 1} {d['message'].splitlines()[0]}"
        for d in errors)

    revealed = []
    for diagnostic in report['generalDiagnostics']:
        message = diagnostic['message']
        if diagnostic['severity'] == 'information' and ' is "' in message:
            revealed.append(message.rsplit(' is "', 1)[1].rstrip('"'))

    return revealed


@pytest.fixture(scope='module')
def revealed(tmp_path_factory):
    return run_pyright(tmp_path_factory.mktemp('typing'))


def test_every_expression_was_reported(revealed):
    assert len(revealed) == len(EXPECTED), revealed


@pytest.mark.parametrize('position,expected', list(enumerate(EXPECTED)))
def test_inferred_type(revealed, position, expected):
    """pandas ships no stubs, so its two are matched loosely - the entarchy
    ones are the claim being made here."""
    actual = revealed[position]

    if expected in ('Series', 'DataFrame'):
        assert expected in actual
    else:
        assert actual == expected
