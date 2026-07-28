import numpy as np
import pytest

from conftest import Session, Subject

# Ground truth for the populated fixture:
#   6 sessions, index 0..5, score = index * 1.5, flag = (index % 2 == 0)


def indices_of(collection):
    return sorted(collection['index'].tolist())


class TestBasics:

    def test_len_and_iteration(self, populated):
        sessions = populated.get(Session)
        assert len(sessions) == 6
        assert len(list(sessions)) == 6

    def test_index_access(self, populated):
        sessions = populated.get(Session)
        first = sessions[0]
        last = sessions[-1]
        assert first.uuid != last.uuid
        assert sessions[len(sessions) - 1].uuid == last.uuid

    def test_dataframe_of_explicit(self, populated):
        df = populated.get(Session).dataframe_of(['index', 'score'])
        assert df.shape == (6, 2)
        assert sorted(df['index'].tolist()) == [0, 1, 2, 3, 4, 5]

    def test_dataframe_of_default_all(self, populated):
        df = populated.get(Session).dataframe_of()
        assert len(df) == 6
        for col in ('id', 'index', 'score', 'flag'):
            assert col in df.columns

    def test_uuid_column(self, populated):
        sessions = populated.get(Session)
        uuids = sessions['uuid']
        assert len(uuids) == 6
        assert set(uuids) == set(sessions.index)

    def test_missing_attribute_raises(self, populated):
        with pytest.raises(AttributeError):
            populated.get(Session)['does_not_exist']


class TestSlicing:

    def test_full_and_stepped(self, populated):
        sessions = populated.get(Session)
        assert len(sessions[:]) == 6
        assert len(sessions[::2]) == 3
        assert len(sessions[1:4]) == 3
        assert sessions[:0] == []

    def test_negative_step(self, populated):
        sessions = populated.get(Session)
        forward = [e.uuid for e in sessions[:]]
        backward = [e.uuid for e in sessions[::-1]]
        assert backward == forward[::-1]
        assert [e.uuid for e in sessions[::-2]] == forward[::-1][::2]

    def test_negative_step_subrange(self, populated):
        sessions = populated.get(Session)
        forward = [e.uuid for e in sessions[:]]
        assert [e.uuid for e in sessions[4:1:-1]] == forward[4:1:-1]


class TestFiltering:

    def test_simple_comparisons(self, populated):
        assert indices_of(populated.get(Session, 'index > 3')) == [4, 5]
        assert indices_of(populated.get(Session, 'index <= 1')) == [0, 1]
        assert indices_of(populated.get(Session, 'score == 3.0')) == [2]
        assert indices_of(populated.get(Session, 'flag == True')) == [0, 2, 4]

    def test_not_equal(self, populated):
        assert indices_of(populated.get(Session, 'index != 3')) == [0, 1, 2, 4, 5]

    def test_in_operator(self, populated):
        assert indices_of(populated.get(Session, 'index IN (1, 3, 5)')) == [1, 3, 5]

    def test_mixed_and_or_precedence(self, populated):
        # (index <= 1 AND flag) OR index >= 4  ->  {0} | {4, 5}
        result = indices_of(populated.get(Session, 'index <= 1 AND flag == True OR index >= 4'))
        assert result == [0, 4, 5]

    def test_xor(self, populated):
        # {0,1,2} XOR {0,2,4} -> {1,4}
        assert indices_of(populated.get(Session, 'index <= 2 XOR flag == True')) == [1, 4]

    def test_not(self, populated):
        assert indices_of(populated.get(Session, 'NOT index > 3')) == [0, 1, 2, 3]

    def test_exist(self, populated):
        sessions = populated.get(Session)
        first = sessions[0]
        first['only_here'] = 1
        assert indices_of(populated.get(Session, 'EXIST(only_here)')) == [first['index']]
        assert len(populated.get(Session, 'NOT(EXIST(only_here))')) == 5

    def test_equality_kwargs(self, populated):
        assert indices_of(populated.get(Session, index=2)) == [2]
        assert indices_of(populated.get(Session, 'index > 0', flag=True)) == [2, 4]

    def test_lowercase_keywords(self, populated):
        assert indices_of(populated.get(Session, 'index > 1 and index < 4')) == [2, 3]

    def test_multiple_expressions_with_or_are_isolated(self, populated):
        # Each expression must act as its own conjunct:
        # (index == 0 OR index == 5) AND (flag == True) -> {0}
        result = indices_of(populated.get(Session, 'index == 0 OR index == 5', 'flag == True'))
        assert result == [0]


class TestParentTraversal:

    def test_explicit_parent_type(self, populated):
        assert len(populated.get(Session, '[Subject]strain == "wildtype"')) == 6
        assert len(populated.get(Session, '[Subject]strain == "mutant"')) == 0

    def test_relative_parent(self, populated):
        assert len(populated.get(Session, '../strain == "wildtype"')) == 6

    def test_parent_attribute_in_dataframe(self, populated):
        df = populated.get(Session).dataframe_of(['index', '../strain'])
        assert set(df['../strain']) == {'wildtype'}


class TestSetOperations:

    def test_intersection_and_union(self, populated):
        low = populated.get(Session, 'index <= 2')
        even = populated.get(Session, 'flag == True')
        assert indices_of(low & even) == [0, 2]
        assert indices_of(low | even) == [0, 1, 2, 4]
        assert indices_of(low - even) == [1]
        assert indices_of(low ^ even) == [1, 4]

    def test_where(self, populated):
        assert indices_of(populated.get(Session).where('index > 3')) == [4, 5]
        assert indices_of(populated.get(Session, 'index > 1').where('flag == True')) == [2, 4]


class TestCollectionSetitem:

    def test_scalar_broadcast(self, populated):
        sessions = populated.get(Session)
        sessions['label'] = 'x'
        assert list(populated.get(Session)['label']) == ['x'] * 6

    def test_identity_attributes_protected(self, populated):
        sessions = populated.get(Session)
        with pytest.raises(RuntimeError):
            sessions['id'] = 'clobbered'
        with pytest.raises(RuntimeError):
            sessions['uuid'] = 'clobbered'
