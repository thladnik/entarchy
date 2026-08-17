"""Collection.sort(): the order a collection is read in.

The order is worked out in Python rather than as a SQL ORDER BY, so that it is
the same whichever backend holds the entarchy - which these check by asserting
the order itself, not merely that some order was applied.
"""
import numpy as np
import pandas as pd
import pytest

import _mp_worker
from conftest import Animal, DeepArchy, Layer, Recording, Roi
from entarchy.backend import SQLiteBackend
from entarchy.core.entity import _attribute_name_of, _natural_sort_key


@pytest.fixture()
def sortable(tmp_path):
    """Two layers of twelve ROIs, so that lexicographic and natural order differ.

    Scores are deliberately not monotonic in the index, and repeat across the
    two layers, so that a sort by score has ties to break.
    """
    base = (tmp_path / 'sortable').as_posix()
    ent = DeepArchy.create(base, SQLiteBackend(base, dbname='sortable.db'))

    with ent:
        animal = Animal(ent, _id='animal_1', _parent=ent.root)
        ent.add_new_entity(animal)
        animal['strain'] = 'wildtype'

        recording = Recording(ent, _id='rec_0', _parent=animal)
        ent.add_new_entity(recording)
        recording['rate'] = 10.0

        for layer_index in range(2):
            layer = Layer(ent, _id=f'plane{layer_index}', _parent=recording)
            ent.add_new_entity(layer)
            layer['depth'] = float(layer_index * 15)

            for roi_index in range(12):
                roi = Roi(ent, _id=f'Roi_{roi_index}', _parent=layer)
                ent.add_new_entity(roi)
                roi['index'] = roi_index
                roi['score'] = float((roi_index * 5) % 12)
                # Only some ROIs have this one, for the missing= tests
                if roi_index % 3 == 0:
                    roi['sparse'] = float(roi_index)
                roi['trace'] = np.arange(roi_index + 1, dtype=float)

    yield ent
    ent.backend.close()


def indices(collection):
    return [roi['index'] for roi in collection]


class TestNaturalKeyHelper:

    @pytest.mark.parametrize('values,expected', [
        (['Roi_10', 'Roi_2', 'Roi_1'], ['Roi_1', 'Roi_2', 'Roi_10']),
        (['plane10', 'plane2', 'plane0'], ['plane0', 'plane2', 'plane10']),
        (['a1b10', 'a1b2', 'a0b99'], ['a0b99', 'a1b2', 'a1b10']),
    ])
    def test_orders_digit_runs_as_numbers(self, values, expected):
        assert sorted(values, key=_natural_sort_key) == expected

    def test_a_shape_mismatch_does_not_raise(self):
        """A tuple of alternating text and numbers would compare int against
        str here; a padded string cannot."""
        assert sorted(['Roi_2b', 'Roi_2', 'Roi_10'], key=_natural_sort_key) == [
            'Roi_2', 'Roi_2b', 'Roi_10']

    def test_text_without_digits_is_unchanged(self):
        assert _natural_sort_key('plane') == 'plane'


class TestAttributeNameHelper:

    @pytest.mark.parametrize('key,expected', [
        ('depth', 'depth'),
        ('../depth', 'depth'),
        ('../../rate', 'rate'),
        ('[Layer]depth', 'depth'),
        ('[Animal]strain', 'strain'),
        ('s2p/npix', 's2p/npix'),
    ])
    def test_strips_parent_addressing(self, key, expected):
        assert _attribute_name_of(key) == expected


class TestSortOrder:

    def test_ascending(self, sortable):
        assert indices(sortable.get(Roi).sort('index')) == sorted(list(range(12)) * 2)

    def test_descending(self, sortable):
        assert indices(sortable.get(Roi).sort('-index')) == sorted(
            list(range(12)) * 2, reverse=True)

    def test_two_keys(self, sortable):
        rois = sortable.get(Roi).sort('-score', 'index')

        frame = rois.dataframe_of(['score', 'index'])
        scores = list(frame['score'])
        assert scores == sorted(scores, reverse=True)

        # Within one score, index ascends
        for score in set(scores):
            block = frame[frame['score'] == score]
            assert list(block['index']) == sorted(block['index'])

    def test_parent_attribute_as_key(self, sortable):
        rois = sortable.get(Roi).sort('-[Layer]depth', 'index')

        depths = [roi.parent['depth'] for roi in rois]
        assert depths == [15.0] * 12 + [0.0] * 12
        assert indices(rois)[:12] == list(range(12))

    def test_relative_parent_attribute_as_key(self, sortable):
        by_explicit = [r.uuid for r in sortable.get(Roi).sort('[Layer]depth', 'index')]
        by_relative = [r.uuid for r in sortable.get(Roi).sort('../depth', 'index')]
        assert by_explicit == by_relative

    def test_length_is_unaffected(self, sortable):
        assert len(sortable.get(Roi).sort('index')) == len(sortable.get(Roi)) == 24


class TestNaturalOrder:

    def test_lexicographic_by_default(self, sortable):
        ids = [roi.id for roi in sortable.get(Roi).sort('id')]
        assert ids[:6] == ['Roi_0', 'Roi_0', 'Roi_1', 'Roi_1', 'Roi_10', 'Roi_10']

    def test_natural_reads_digits_as_numbers(self, sortable):
        ids = [roi.id for roi in sortable.get(Roi).sort('id', natural=True)]
        assert ids[:6] == ['Roi_0', 'Roi_0', 'Roi_1', 'Roi_1', 'Roi_2', 'Roi_2']
        assert ids[-2:] == ['Roi_11', 'Roi_11']

    def test_numeric_keys_are_left_alone(self, sortable):
        """Padding the digits of a float column would sort 10.0 before 9.0."""
        plain = indices(sortable.get(Roi).sort('index'))
        natural = indices(sortable.get(Roi).sort('index', natural=True))
        assert plain == natural == sorted(list(range(12)) * 2)


class TestTies:

    def test_uuid_breaks_ties_reproducibly(self, sortable):
        """Every ROI shares its score with one in the other layer, so the whole
        collection is ties."""
        first = [r.uuid for r in sortable.get(Roi).sort('score')]
        second = [r.uuid for r in sortable.get(Roi).sort('score')]
        assert first == second

    def test_the_tiebreaker_is_the_uuid(self, sortable):
        rois = sortable.get(Roi).sort('score')
        frame = rois.dataframe_of(['score'])

        for score in set(frame['score']):
            block = frame[frame['score'] == score]
            assert list(block.index) == sorted(block.index)


class TestMissing:

    def test_missing_goes_last_by_default(self, sortable):
        frame = sortable.get(Roi).sort('sparse').dataframe_of(['sparse'])
        values = list(frame['sparse'])

        present = [v for v in values if not pd.isna(v)]
        assert present == sorted(present)
        assert all(pd.isna(v) for v in values[len(present):])

    def test_missing_can_go_first(self, sortable):
        frame = sortable.get(Roi).sort('sparse', missing='first').dataframe_of(['sparse'])
        values = list(frame['sparse'])

        absent = [v for v in values if pd.isna(v)]
        assert all(pd.isna(v) for v in values[:len(absent)])

    def test_missing_is_still_last_when_descending(self, sortable):
        frame = sortable.get(Roi).sort('-sparse').dataframe_of(['sparse'])
        values = list(frame['sparse'])

        present = [v for v in values if not pd.isna(v)]
        assert present == sorted(present, reverse=True)
        assert all(pd.isna(v) for v in values[len(present):])

    def test_a_bad_missing_value_is_refused(self, sortable):
        with pytest.raises(ValueError, match='must be "first" or "last"'):
            sortable.get(Roi).sort('index', missing='middle')


class TestRefusals:

    def test_a_blob_cannot_be_a_sort_key(self, sortable):
        with pytest.raises(TypeError, match='stored as blobs'):
            sortable.get(Roi).sort('trace')

    def test_an_unknown_attribute_is_refused(self, sortable):
        with pytest.raises(AttributeError, match='not stored on any entity'):
            sortable.get(Roi).sort('no_such_thing')

    def test_no_keys_is_refused(self, sortable):
        with pytest.raises(ValueError, match='at least one attribute name'):
            sortable.get(Roi).sort()

    def test_a_non_string_key_is_refused(self, sortable):
        with pytest.raises(TypeError, match='attribute names'):
            sortable.get(Roi).sort(3)

    def test_a_bare_minus_is_refused(self, sortable):
        with pytest.raises(ValueError, match='empty sort key'):
            sortable.get(Roi).sort('-')

    def test_the_refusal_happens_before_any_reading(self, sortable):
        """sort() should complain about the line that is wrong, not later from
        inside whatever first happened to read the collection."""
        with pytest.raises(TypeError):
            sortable.get(Roi).sort('trace')


class TestAccessPaths:

    def test_indexing(self, sortable):
        rois = sortable.get(Roi).sort('index')
        assert rois[0]['index'] == 0
        assert rois[-1]['index'] == 11
        assert rois[5]['index'] == indices(rois)[5]

    def test_out_of_range_index(self, sortable):
        rois = sortable.get(Roi).sort('index')
        with pytest.raises(IndexError):
            rois[24]

    def test_slicing(self, sortable):
        rois = sortable.get(Roi).sort('index')
        assert [r['index'] for r in rois[:4]] == [0, 0, 1, 1]
        assert [r['index'] for r in rois[2:6]] == [1, 1, 2, 2]
        assert [r['index'] for r in rois[::2]] == list(range(12))

    def test_iteration(self, sortable):
        assert indices(sortable.get(Roi).sort('-index'))[:2] == [11, 11]

    def test_dataframe_follows_the_order(self, sortable):
        rois = sortable.get(Roi).sort('-index')
        frame = rois.dataframe_of(['index'])

        assert list(frame['index']) == indices(rois)
        assert list(frame.index) == [roi.uuid for roi in rois]

    def test_dataframe_with_parent_columns_follows_the_order(self, sortable):
        rois = sortable.get(Roi).sort('-[Layer]depth', 'index')
        frame = rois.dataframe_of(['index', '[Layer]depth'])

        assert list(frame['[Layer]depth']) == [15.0] * 12 + [0.0] * 12
        # And the values still belong to the entities they are about
        for roi in rois:
            assert frame.loc[roi.uuid, '[Layer]depth'] == roi.parent['depth']

    def test_preview_follows_the_order(self, sortable):
        rois = sortable.get(Roi).sort('-index')
        assert list(rois.preview(4, ['index'])['index']) == [11, 11, 10, 10]

    def test_map_follows_the_order(self, sortable):
        rois = sortable.get(Roi).sort('-index')
        assert rois.map(lambda roi: roi['index']) == indices(rois)

    def test_repr_html_does_not_break(self, sortable):
        assert 'Roi' in sortable.get(Roi).sort('index')._repr_html_()


class TestDerivation:

    def test_the_original_is_unchanged(self, sortable):
        rois = sortable.get(Roi)
        rois.sort('index')
        assert not rois.is_sorted

    def test_sort_state_is_reported(self, sortable):
        rois = sortable.get(Roi).sort('[Layer]depth', '-index')
        assert rois.is_sorted
        assert rois.sort_keys == ['[Layer]depth', '-index']

    def test_an_unsorted_collection_reports_no_keys(self, sortable):
        assert sortable.get(Roi).sort_keys == []
        assert not sortable.get(Roi).is_sorted

    def test_where_drops_the_sort(self, sortable):
        narrowed = sortable.get(Roi).sort('index').where('index > 3')
        assert not narrowed.is_sorted
        assert len(narrowed) == 16

    def test_set_operations_drop_the_sort(self, sortable):
        # A filter rather than every Roi, because union and difference refuse a
        # universal set - unrelated to sorting, but it has to be a real set
        rois = sortable.get(Roi, 'index >= 0').sort('index')
        assert rois.is_sorted

        assert not (rois & 'index > 3').is_sorted
        assert not (rois | 'index > 3').is_sorted
        assert not (rois - 'index > 3').is_sorted
        assert not (rois ^ 'index > 3').is_sorted
        assert not (~rois).is_sorted

    def test_sorting_again_replaces_the_order(self, sortable):
        rois = sortable.get(Roi).sort('index').sort('-index')
        assert rois.sort_keys == ['-index']
        assert indices(rois)[:2] == [11, 11]

    def test_a_sorted_collection_can_be_filtered_then_sorted(self, sortable):
        rois = sortable.get(Roi).where('index > 8').sort('-index')
        assert indices(rois) == [11, 11, 10, 10, 9, 9]


class TestSortedCollectionWrites:

    def test_writing_through_a_sorted_collection_lands_correctly(self, sortable):
        rois = sortable.get(Roi).sort('-index')
        frame = rois.dataframe_of(['index'])

        rois['doubled'] = frame['index'] * 2
        sortable.commit()

        for roi in sortable.get(Roi):
            assert roi['doubled'] == roi['index'] * 2


class TestMapAsyncInteraction:

    def test_locality_is_off_by_default_when_sorted(self, sortable):
        """Grouping by parent would undo the order that was asked for."""
        rois = sortable.get(Roi).sort('-index')
        rows = rois._rows()
        assert [uuid for uuid, _ in rows] == [roi.uuid for roi in rois]

    def test_map_async_processes_in_order(self, sortable):
        rois = sortable.get(Roi).sort('-index')
        rois.map_async(_mp_worker.record_order, _worker_num=1, _calibrate=False)

        frame = rois.dataframe_of(['index', 'processed_at'])
        assert list(frame['processed_at']) == sorted(frame['processed_at'])
        assert list(frame['index']) == indices(rois)
