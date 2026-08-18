"""Tab completion for stored names, and the protocol it rides on."""
import numpy as np
import pytest

from conftest import Animal, DeepArchy, Layer, Recording, Roi
from entarchy.backend import SQLiteBackend
from entarchy.core.entity import CollectionIterator


@pytest.fixture()
def filled(tmp_path):
    base = (tmp_path / 'completing').as_posix()
    ent = DeepArchy.create(base, SQLiteBackend(base, dbname='completing.db'))

    with ent:
        animal = Animal(ent, _id='animal_1', _parent=ent.root)
        ent.add_new_entity(animal)

        recording = Recording(ent, _id='rec_0', _parent=animal)
        ent.add_new_entity(recording)

        layer = Layer(ent, _id='plane0', _parent=recording)
        ent.add_new_entity(layer)

        for index in range(3):
            roi = Roi(ent, _id=f'Roi_{index}', _parent=layer)
            ent.add_new_entity(roi)
            roi['index'] = index
            roi['s2p/npix'] = 10 + index
            roi['dff'] = np.arange(5.0)

    yield ent
    ent.backend.close()


class TestKeyCompletion:
    """What IPython asks an object for when the cursor is inside a subscript."""

    def test_an_entity_offers_the_names_it_stores(self, filled):
        roi = filled.get(Roi, 'index == 0')[0]
        offered = roi._ipython_key_completions_()

        assert 's2p/npix' in offered
        assert 'dff' in offered
        assert offered == roi.keys()

    def test_a_collection_offers_its_columns(self, filled):
        rois = filled.get(Roi)
        assert rois._ipython_key_completions_() == list(rois.columns)

    def test_a_description_offers_its_sections(self, filled):
        description = filled.get(Roi)[0].describe()
        offered = description._ipython_key_completions_()

        assert 'attributes' in offered
        assert all(name in description for name in offered)

    def test_a_stored_name_is_not_reachable_any_other_way(self, filled):
        """Which is why the hook has to exist: `s2p/npix` is a row in the
        attributes table, so dir() cannot see it and no static tool can."""
        roi = filled.get(Roi, 'index == 0')[0]

        assert 's2p/npix' in roi.keys()
        assert 's2p/npix' not in dir(roi)

    def test_a_completer_never_raises(self, filled, monkeypatch):
        """It runs on a keystroke. An entity whose backend has gone away should
        still be typeable rather than raising into the prompt."""
        roi = filled.get(Roi, 'index == 0')[0]

        def explode(*args, **kwargs):
            raise RuntimeError('backend is gone')

        monkeypatch.setattr(type(roi), 'keys', explode)
        assert roi._ipython_key_completions_() == []


class TestIPythonActuallyUsesIt:
    """The whole path, since the hook is only worth anything if IPython finds
    it. Without one, the completer falls back to the global namespace and
    offers every builtin and magic there is."""

    @pytest.fixture()
    def completer(self, filled):
        pytest.importorskip('IPython')
        from IPython.terminal.interactiveshell import TerminalInteractiveShell

        shell = TerminalInteractiveShell.instance()
        shell.user_ns.update({'roi': filled.get(Roi, 'index == 0')[0],
                              'rois': filled.get(Roi)})
        shell.Completer.use_jedi = False

        return shell.Completer

    def _completions(self, completer, line):
        from IPython.core.completer import provisionalcompleter

        with provisionalcompleter():
            return [match.text for match in completer.completions(line, len(line))]

    def test_an_entity_completes_to_its_own_names(self, completer, filled):
        offered = self._completions(completer, "roi['")

        assert set(offered) == set(filled.get(Roi, 'index == 0')[0].keys())
        assert 'ArithmeticError' not in offered

    def test_a_collection_completes_to_its_own_names(self, completer, filled):
        offered = self._completions(completer, "rois['")

        assert set(offered) == set(filled.get(Roi).columns)
        assert 'ArithmeticError' not in offered


class TestIterationProtocol:

    def test_the_iterator_is_its_own_iterable(self, filled):
        """A for-loop only needs __next__, so this went unnoticed - but an
        iterator that cannot answer iter() is not one, and a type checker is
        right to say a collection built on it is not iterable."""
        iterator = iter(filled.get(Roi))

        assert isinstance(iterator, CollectionIterator)
        assert iter(iterator) is iterator

    def test_it_still_walks_the_collection_once(self, filled):
        iterator = iter(filled.get(Roi))
        walked = [entity.id for entity in iterator]

        assert sorted(walked) == ['Roi_0', 'Roi_1', 'Roi_2']
        assert list(iterator) == []
