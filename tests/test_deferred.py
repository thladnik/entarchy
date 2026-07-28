"""DeferredEntityCollection: query expressions built before binding to an entarchy."""
import pytest

from conftest import Layer, Roi, Session, Subject
from entarchy.core.entity import Collection, DeferredEntityCollection


def indices_of(collection):
    return sorted(collection['index'].tolist())


class TestConstruction:

    def test_entity_constructor_with_string_returns_deferred(self):
        obj = Session('index > 2')
        assert isinstance(obj, DeferredEntityCollection)
        assert obj.entity_type is Session
        assert obj.expression == 'index > 2'

    def test_repr(self):
        assert 'Session' in repr(Session('index > 2'))

    def test_as_tree_matches_direct_parse(self):
        from entarchy.core.query import parse_boolean_expression
        assert Session('index > 2').as_tree == parse_boolean_expression('index > 2')


class TestAlgebra:

    def test_and(self, populated):
        deferred = Session('index > 1') & 'flag == True'
        assert indices_of(deferred.get_from(populated)) == [2, 4]

    def test_or(self, populated):
        deferred = Session('index == 0') | 'index == 5'
        assert indices_of(deferred.get_from(populated)) == [0, 5]

    def test_invert(self, populated):
        deferred = ~Session('index > 1')
        assert indices_of(deferred.get_from(populated)) == [0, 1]

    def test_sub(self, populated):
        deferred = Session('index > 1') - 'flag == True'
        assert indices_of(deferred.get_from(populated)) == [3, 5]

    def test_chained(self, populated):
        deferred = (Session('index > 0') & 'index < 5') - 'flag == True'
        assert indices_of(deferred.get_from(populated)) == [1, 3]

    def test_precedence_is_preserved_when_combining(self, populated):
        """Sub-expressions must stay grouped when combined."""
        deferred = Session('index == 0 OR index == 5') & 'flag == True'
        # (0 or 5) and even -> {0}, not 0 or (5 and even) -> {0}
        assert indices_of(deferred.get_from(populated)) == [0]

    def test_get_from_returns_custom_collection_type(self, populated):
        result = Session('index > 1').get_from(populated)
        assert isinstance(result, Collection)
        assert result.entity_type is Session


class TestCrossTypeExpressions:

    def test_other_entity_type_is_prefixed(self):
        combined = Roi('index == 0') & Layer('depth == 0.0')
        assert '[Layer]depth == 0.0' in combined.expression

    def test_same_entity_type_is_not_prefixed(self):
        combined = Roi('index == 0') & Roi('index == 1')
        assert '[Roi]' not in combined.expression

    def test_cross_type_query_runs(self, deep):
        combined = Roi('index == 0') & Layer('depth == 0.0')
        result = combined.get_from(deep)
        assert len(result) == 2  # one per recording


class TestCollectionSetOperationGuards:
    """Set operations need a bounded operand - the universal set is rejected."""

    def test_union_with_unfiltered_raises(self, populated):
        everything = populated.get(Session)
        filtered = populated.get(Session, 'index > 1')
        with pytest.raises(RuntimeError, match='universal set'):
            everything | filtered

    def test_intersection_with_unfiltered_raises(self, populated):
        everything = populated.get(Session)
        filtered = populated.get(Session, 'index > 1')
        with pytest.raises(RuntimeError, match='universal set'):
            everything & filtered

    def test_complement_of_unfiltered_raises(self, populated):
        with pytest.raises(RuntimeError, match='universal set'):
            ~populated.get(Session)

    def test_type_mismatch_raises(self, populated):
        sessions = populated.get(Session, 'index > 1')
        subjects = populated.get(Subject, 'strain == "wildtype"')
        with pytest.raises(ValueError, match='same entity type'):
            sessions | subjects

    def test_where_on_unfiltered_collection_is_allowed(self, populated):
        # where() intentionally supports narrowing the universal set
        assert indices_of(populated.get(Session).where('index > 3')) == [4, 5]
