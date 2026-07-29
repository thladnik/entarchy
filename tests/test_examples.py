"""Execute the example notebooks, so they cannot silently go stale."""
import pathlib

import pytest

nbformat = pytest.importorskip('nbformat', reason='notebook execution requires nbformat')
nbclient = pytest.importorskip('nbclient', reason='notebook execution requires nbclient')

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / 'examples'
NOTEBOOKS = sorted(EXAMPLES.glob('*.ipynb'))


def test_examples_exist():
    assert NOTEBOOKS, f'no example notebooks found in {EXAMPLES}'


@pytest.mark.parametrize('notebook_path', NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_stored_outputs(notebook_path):
    """Outputs make diffs unreadable, so the committed notebooks stay clean."""
    notebook = nbformat.read(notebook_path, as_version=4)

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            assert not cell.get('outputs'), f'cell {index} has stored outputs'
            assert cell.get('execution_count') is None, f'cell {index} has an execution count'


@pytest.mark.slow
@pytest.mark.parametrize('notebook_path', NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(notebook_path, tmp_path):
    """Run every cell; any exception fails the test."""
    from nbclient import NotebookClient

    notebook = nbformat.read(notebook_path, as_version=4)

    client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                            resources={'metadata': {'path': str(tmp_path)}})
    client.execute()
