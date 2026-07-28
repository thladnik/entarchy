"""Multi-level hierarchy: parent traversal in filters, dataframes and helpers."""
import pytest

from conftest import Animal, Layer, Recording, Roi
from entarchy.core.entity import _find_path, get_ancestor_distance_from_nested

HIERARCHY = {'Animal': {'Recording': {'Layer': {'Roi': {}}}}}


class TestAncestorDistanceHelper:

    @pytest.mark.parametrize('descendant,ancestor,expected', [
        ('Roi', 'Roi', 0),
        ('Roi', 'Layer', 1),
        ('Roi', 'Recording', 2),
        ('Roi', 'Animal', 3),
        ('Layer', 'Animal', 2),
        ('Recording', 'Animal', 1),
    ])
    def test_distances(self, descendant, ancestor, expected):
        assert get_ancestor_distance_from_nested(HIERARCHY, descendant, ancestor) == expected

    @pytest.mark.parametrize('descendant,ancestor', [
        ('Animal', 'Roi'),      # wrong direction
        ('Layer', 'Nonexistent'),
        ('Nonexistent', 'Animal'),
    ])
    def test_non_ancestors_return_none(self, descendant, ancestor):
        assert get_ancestor_distance_from_nested(HIERARCHY, descendant, ancestor) is None

    def test_find_path(self):
        assert _find_path(HIERARCHY, 'Roi') == ['Animal', 'Recording', 'Layer', 'Roi']
        assert _find_path(HIERARCHY, 'Animal') == ['Animal']
        assert _find_path(HIERARCHY, 'Nope') is None


class TestBackendAncestorDistance:

    def test_matches_helper(self, deep):
        from entarchy.backend.sqlite import _get_entity_type_ancestor_distance

        with deep.backend.sql_session as session:
            assert _get_entity_type_ancestor_distance(session, 'Roi', 'Animal') == 3
            assert _get_entity_type_ancestor_distance(session, 'Roi', 'Layer') == 1
            assert _get_entity_type_ancestor_distance(session, 'Roi', 'Roi') == 0
            assert _get_entity_type_ancestor_distance(session, 'Animal', 'Roi') is None


class TestStructure:

    def test_entity_counts(self, deep):
        assert len(deep.get(Animal)) == 1
        assert len(deep.get(Recording)) == 2
        assert len(deep.get(Layer)) == 4
        assert len(deep.get(Roi)) == 12

    def test_parent_chain(self, deep):
        roi = deep.get(Roi)[0]
        layer = roi.parent
        recording = layer.parent
        animal = recording.parent

        assert isinstance(layer, Layer)
        assert isinstance(recording, Recording)
        assert isinstance(animal, Animal)
        assert animal.parent is deep.root or animal.parent.uuid == deep.root.uuid

    def test_path_property(self, deep):
        roi = deep.get(Roi, 'index == 0')[0]
        # Path is relative to the root entarchy entity, which is excluded
        assert roi.path.count('/') == 3
        assert roi.path.startswith('animal_1/')
        assert roi.path.endswith('/roi_0')

    def test_root_has_no_parent(self, deep):
        assert deep.root.parent is None


class TestParentFilters:

    def test_explicit_type_two_levels_up(self, deep):
        assert len(deep.get(Roi, '[Recording]rate == 10.0')) == 6

    def test_explicit_type_three_levels_up(self, deep):
        assert len(deep.get(Roi, '[Animal]strain == "wildtype"')) == 12
        assert len(deep.get(Roi, '[Animal]strain == "mutant"')) == 0

    def test_relative_two_levels_up(self, deep):
        assert len(deep.get(Roi, '../../rate == 10.0')) == 6

    def test_relative_three_levels_up(self, deep):
        assert len(deep.get(Roi, '../../../age == 12')) == 12

    def test_relative_and_explicit_agree(self, deep):
        relative = len(deep.get(Roi, '../depth == 15.0'))
        explicit = len(deep.get(Roi, '[Layer]depth == 15.0'))
        assert relative == explicit == 6

    def test_combined_with_own_attribute(self, deep):
        result = deep.get(Roi, 'index == 0 AND [Recording]rate == 11.0')
        assert len(result) == 2  # one per layer of that recording

    def test_non_ancestor_type_raises(self, deep):
        with pytest.raises(ValueError, match='not an ancestor'):
            len(deep.get(Layer, '[Roi]index == 0'))


class TestParentAttributesInDataFrame:

    def test_multi_level_columns(self, deep):
        df = deep.get(Roi).dataframe_of(['index', '../depth', '../../rate', '[Animal]strain'])
        assert len(df) == 12
        assert set(df['[Animal]strain']) == {'wildtype'}
        assert set(df['../depth']) == {0.0, 15.0}
        assert set(df['../../rate']) == {10.0, 11.0}

    def test_column_order_preserved(self, deep):
        cols = ['[Animal]strain', 'index', '../depth']
        df = deep.get(Roi).dataframe_of(cols)
        assert list(df.columns) == cols

    def test_parent_attributes_cannot_be_written(self, deep):
        import pandas as pd

        rois = deep.get(Roi)
        df = pd.DataFrame(index=rois.index, columns=['../depth'], data=[1.0] * len(rois))
        with pytest.raises(RuntimeError, match='parent attributes'):
            rois.update(df)
