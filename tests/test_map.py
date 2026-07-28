import pytest

import _mp_worker
from conftest import Session


class TestMap:

    def test_sequential_map_returns_results(self, populated):
        results = populated.get(Session).map(lambda e: e['index'] * 10)
        assert sorted(results) == [0, 10, 20, 30, 40, 50]

    def test_sequential_map_with_writes(self, populated):
        populated.get(Session).map(_mp_worker.double_score)
        out = populated.get(Session)[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()


@pytest.mark.slow
class TestMapAsync:

    def test_map_async_multiprocessing(self, populated):
        populated.get(Session).map_async(_mp_worker.double_score, _worker_num=2)

        out = populated.get(Session)[['score', 'doubled']]
        assert len(out) == 6
        assert (out['doubled'] == out['score'] * 2).all()
