"""Notebook support: interactive definitions, and the rich representations."""
import pickle
import sys
import types

import pandas as pd
import pytest

import _mp_worker
from conftest import Recording, Roi, Session, Subject
from entarchy.core import entity as entity_module
from entarchy.core.entity import shutdown_worker_pool


@pytest.fixture(autouse=True)
def _release_pool():
    yield
    shutdown_worker_pool()


@pytest.fixture()
def many(ent):
    with ent:
        subject = Subject(ent, _id='subject', _parent=ent.root)
        ent.add_new_entity(subject)
        for i in range(8):
            session = Session(ent, _id=f'sess_{i}', _parent=subject)
            ent.add_new_entity(session)
            session['index'] = i
            session['score'] = float(i)
    return ent


class TestInteractiveDetection:

    def test_module_objects_are_not_interactive(self):
        assert not entity_module._defined_interactively(_mp_worker.double_score)
        assert not entity_module._defined_interactively(Session)

    def test_main_with_a_file_is_not_interactive(self, monkeypatch):
        fake_main = types.ModuleType('__main__')
        fake_main.__file__ = 'script.py'
        monkeypatch.setitem(sys.modules, '__main__', fake_main)

        def cell_function(entity):
            pass

        cell_function.__module__ = '__main__'
        assert not entity_module._defined_interactively(cell_function)

    def test_main_without_a_file_is_interactive(self, monkeypatch):
        """A Jupyter kernel's __main__ has no __file__ and cannot be imported."""
        fake_main = types.ModuleType('__main__')
        monkeypatch.setitem(sys.modules, '__main__', fake_main)

        def cell_function(entity):
            pass

        cell_function.__module__ = '__main__'
        assert entity_module._defined_interactively(cell_function)

    def test_instances_are_judged_by_their_class(self, monkeypatch, ent):
        fake_main = types.ModuleType('__main__')
        monkeypatch.setitem(sys.modules, '__main__', fake_main)

        monkeypatch.setattr(type(ent), '__module__', '__main__')
        assert entity_module._defined_interactively(ent)


class TestByValue:

    def test_roundtrip_restores_the_object(self):
        pytest.importorskip('cloudpickle')

        wrapped = entity_module._ByValue({'a': 1, 'b': [2, 3]})
        assert pickle.loads(pickle.dumps(wrapped)) == {'a': 1, 'b': [2, 3]}

    def test_serialises_once_in_the_parent(self):
        pytest.importorskip('cloudpickle')

        wrapped = entity_module._ByValue(_mp_worker.double_score)
        first = pickle.dumps(wrapped)
        second = pickle.dumps(wrapped)
        assert first == second

    def test_for_workers_passes_through_when_not_needed(self):
        payload = ('a', 1)
        assert entity_module._for_workers(payload, False) is payload


class TestInteractiveFallback:
    """Without cloudpickle the pool would hang, so the work stays in this process."""

    def test_warns_and_runs_in_process(self, many, capsys, monkeypatch):
        monkeypatch.setattr(entity_module, '_cloudpickle_available', lambda: False)
        monkeypatch.setattr(entity_module, '_defined_interactively',
                            lambda obj: obj is _mp_worker.double_score)

        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2,
                                    _calibrate=False)

        output = capsys.readouterr().out
        assert 'cloudpickle is not installed' in output
        assert 'Running in this process' in output
        assert 'worker pool' not in output

        out = many.get(Session)[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()

    def test_names_the_offending_definitions(self, many, capsys, monkeypatch):
        monkeypatch.setattr(entity_module, '_cloudpickle_available', lambda: False)
        monkeypatch.setattr(entity_module, '_defined_interactively',
                            lambda obj: obj is _mp_worker.double_score)

        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2,
                                    _calibrate=False)
        assert 'double_score' in capsys.readouterr().out

    def test_uses_workers_when_cloudpickle_is_available(self, many, capsys, monkeypatch):
        pytest.importorskip('cloudpickle')
        monkeypatch.setattr(entity_module, '_defined_interactively',
                            lambda obj: obj is _mp_worker.double_score)

        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2,
                                    _calibrate=False)

        assert 'by value' in capsys.readouterr().out
        out = many.get(Session)[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()


class TestEntityHtmlRepr:

    def test_shows_identity_without_reading_values(self, many):
        entity = many.get(Session)[0]
        entity._attribute_cache.clear()

        rendered = entity._repr_html_()

        assert entity.id in rendered
        assert entity.uuid in rendered
        assert 'Session' in rendered
        assert 'index' in rendered and 'score' in rendered
        # Names are listed, values are not loaded
        assert entity._attribute_cache == {}

    def test_shows_the_path(self, many):
        entity = many.get(Session)[0]
        assert 'subject/' in entity._repr_html_()

    def test_attribute_list_is_capped(self, many, monkeypatch):
        monkeypatch.setattr(entity_module, '_HTML_MAX_ATTRIBUTES', 2)
        entity = many.get(Session)[0]
        assert 'more' in entity._repr_html_()

    def test_escapes_html(self, ent):
        with ent:
            subject = Subject(ent, _id='<script>alert(1)</script>', _parent=ent.root)
            ent.add_new_entity(subject)

        rendered = subject._repr_html_()
        assert '<script>' not in rendered
        assert '&lt;script&gt;' in rendered

    def test_never_raises(self, many, monkeypatch):
        entity = many.get(Session)[0]
        monkeypatch.setattr(type(entity), 'keys',
                            lambda self: (_ for _ in ()).throw(RuntimeError('boom')))

        rendered = entity._repr_html_()
        assert 'Session' in rendered


class TestEntityHtmlReprLinks:
    """The links line, which is the only place in the repr that links appear:
    keys() lists attributes, and links are the other half of what an entity has."""

    @pytest.fixture()
    def linked(self, deep):
        recording = sorted(deep.get(Recording), key=lambda e: e.id)[0]
        rois = sorted(deep.get(Roi), key=lambda e: e.uuid)

        with deep:
            for roi in rois[:3]:
                deep.link(recording, roi, 'mean_response', mean_dff=0.1)
            deep.link(recording, rois[0], 'peak_latency', latency=0.3)

        return deep, recording, rois

    def test_omitted_when_the_entity_has_no_links(self, deep):
        """An entarchy that uses no links reads exactly as it did before."""
        rendered = deep.get(Roi)[0]._repr_html_()

        assert 'link kind' not in rendered
        assert 'entity.links(' not in rendered

    def test_lists_kinds_with_counts(self, linked):
        ent, recording, rois = linked

        rendered = recording._repr_html_()

        assert '2 link kinds' in rendered
        assert 'mean_response' in rendered and '(3)' in rendered
        assert 'peak_latency' in rendered and '(1)' in rendered
        assert 'entity.links(' in rendered

    def test_counts_links_from_either_end(self, linked):
        """The ROI is the linked end of every one of these, so a count that only
        looked at linker_uuid would show it as unlinked."""
        ent, recording, rois = linked

        assert '2 link kinds' in rois[0]._repr_html_()

    def test_singular_for_a_single_kind(self, linked):
        ent, recording, rois = linked
        assert '1 link kind<' in rois[1]._repr_html_()

    def test_does_not_load_attribute_values(self, linked):
        ent, recording, rois = linked
        recording._attribute_cache.clear()

        recording._repr_html_()

        assert recording._attribute_cache == {}

    def test_kind_list_is_capped(self, linked, monkeypatch):
        monkeypatch.setattr(entity_module, '_HTML_MAX_LINK_TYPES', 1)
        ent, recording, rois = linked

        assert 'and 1 more' in recording._repr_html_()

    def test_escapes_the_kind_name(self, deep):
        """Kind names are invented at runtime and never validated, so they reach
        the repr as whatever the user typed."""
        recording = sorted(deep.get(Recording), key=lambda e: e.id)[0]
        roi = sorted(deep.get(Roi), key=lambda e: e.uuid)[0]

        with deep:
            deep.link(recording, roi, '<script>alert(1)</script>')

        rendered = recording._repr_html_()
        assert '<script>' not in rendered
        assert '&lt;script&gt;' in rendered

    def test_survives_a_backend_that_cannot_count_links(self, deep, monkeypatch):
        """Links are an addition to the repr, so a backend without them loses the
        line rather than the whole representation."""
        entity = deep.get(Roi)[0]
        monkeypatch.setattr(type(entity), 'link_counts',
                            lambda self: (_ for _ in ()).throw(RuntimeError('boom')))

        rendered = entity._repr_html_()

        assert 'Roi' in rendered and 'index' in rendered
        assert 'link kind' not in rendered


class TestCollectionHtmlRepr:

    def test_shows_type_and_count(self, many):
        rendered = many.get(Session)._repr_html_()
        assert 'Session' in rendered
        assert '8' in rendered

    def test_previews_a_bounded_number_of_entities(self, many, monkeypatch):
        monkeypatch.setattr(entity_module, '_HTML_PREVIEW_ROWS', 2)
        rendered = many.get(Session)._repr_html_()
        assert 'showing 2 of 8' in rendered

    def test_empty_collection(self, many):
        assert 'no matching entities' in many.get(Session, 'index > 99')._repr_html_()

    def test_uses_the_custom_name(self, many):
        collection = many.get(Session)
        collection.rename('My selection')
        assert 'My selection' in collection._repr_html_()

    def test_does_not_load_attribute_values(self, many):
        collection = many.get(Session)
        collection._repr_html_()
        # The cache stays empty; values are only fetched on demand
        assert collection._cache.empty

    def test_never_raises_even_when_repr_fails(self, many, monkeypatch):
        """__repr__ itself queries the backend, so the fallback must tolerate it
        failing too."""
        collection = many.get(Session)
        monkeypatch.setattr(type(collection), '__len__',
                            lambda self: (_ for _ in ()).throw(RuntimeError('boom')))

        rendered = collection._repr_html_()
        assert 'repr failed' in rendered


class TestCollectionPreview:

    def test_returns_a_dataframe_of_the_first_entities(self, many):
        preview = many.get(Session).preview(3)

        assert isinstance(preview, pd.DataFrame)
        assert len(preview) == 3
        assert 'index' in preview.columns

    def test_respects_explicit_attributes(self, many):
        preview = many.get(Session).preview(2, attribute_names=['score'])
        assert list(preview.columns) == ['score']

    def test_empty_collection(self, many):
        assert many.get(Session, 'index > 99').preview().empty
