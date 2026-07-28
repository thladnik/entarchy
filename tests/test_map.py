import multiprocessing

import pytest

import _mp_worker
from conftest import Session, Subject


@pytest.fixture()
def many(ent):
    """A collection large enough to spread over several workers."""
    with ent:
        subject = Subject(ent, _id='subject', _parent=ent.root)
        ent.add_new_entity(subject)
        for i in range(12):
            session = Session(ent, _id=f'sess_{i}', _parent=subject)
            ent.add_new_entity(session)
            session['index'] = i
            session['score'] = float(i)
    return ent


class TestMap:

    def test_sequential_map_returns_results(self, populated):
        results = populated.get(Session).map(lambda e: e['index'] * 10)
        assert sorted(results) == [0, 10, 20, 30, 40, 50]

    def test_sequential_map_with_writes(self, populated):
        populated.get(Session).map(_mp_worker.double_score)
        out = populated.get(Session)[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()


class TestMapAsyncBasics:

    def test_empty_collection_returns_early(self, populated, capsys):
        populated.get(Session, 'index > 100').map_async(_mp_worker.double_score)
        assert 'No entities to operate on' in capsys.readouterr().out

    def test_results_are_written(self, populated):
        populated.get(Session).map_async(_mp_worker.double_score, _worker_num=2)

        out = populated.get(Session)[['score', 'doubled']]
        assert len(out) == 6
        assert (out['doubled'] == out['score'] * 2).all()

    def test_every_entity_is_processed_exactly_once(self, many):
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2, _calibrate=False)

        out = many.get(Session)[['score', 'doubled']]
        assert len(out) == 12
        assert (out['doubled'] == out['score'] * 2).all()

    def test_kwargs_are_forwarded(self, populated):
        populated.get(Session).map_async(_mp_worker.record_device, _worker_num=2,
                                         _calibrate=False)
        assert set(populated.get(Session)['device']) == {'None'}

    def test_failure_propagates(self, many):
        with pytest.raises(Exception):
            many.get(Session).map_async(_mp_worker.double_score, _worker_num=2,
                                        _calibrate=False, unexpected_kwarg=1)


class TestWorkerReuse:
    """Item 1: one Entarchy (and one connection) per worker, not per entity."""

    def test_backend_is_reused_across_tasks(self, many):
        worker_num = 2
        many.get(Session).map_async(_mp_worker.record_worker_state,
                                    _worker_num=worker_num, _calibrate=False)

        state = many.get(Session)[['backend_was_open', 'pid']]

        # Only the very first task of each worker may find no open connection
        cold_starts = (~state['backend_was_open'].astype(bool)).sum()
        assert cold_starts <= worker_num
        assert state['backend_was_open'].astype(bool).sum() >= len(state) - worker_num

    def test_work_is_spread_over_processes(self, many):
        many.get(Session).map_async(_mp_worker.record_worker_state,
                                    _worker_num=2, _calibrate=False)

        pids = set(many.get(Session)['pid'])
        assert len(pids) > 1

    def test_worker_registry_does_not_grow(self, many):
        """Entities are released after processing, so the registry stays small."""
        many.get(Session).map_async(_mp_worker.record_worker_state,
                                    _worker_num=2, _calibrate=False)

        # At most the entity being processed is registered while a task runs
        assert max(many.get(Session)['registry_size']) <= 1


class TestWorkerCountGuard:
    """Item 2: the worker count must stay positive."""

    @pytest.mark.parametrize('worker_num', [0, -4, 1])
    def test_non_positive_worker_num_still_runs(self, populated, worker_num):
        populated.get(Session).map_async(_mp_worker.double_score, _worker_num=worker_num)

        out = populated.get(Session)[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()

    def test_small_machine_cpu_count(self, populated, monkeypatch):
        """cpu_count() - 2 is <= 0 on one and two core machines."""
        monkeypatch.setattr(multiprocessing, 'cpu_count', lambda: 1)

        populated.get(Session).map_async(_mp_worker.double_score, _calibrate=False)

        out = populated.get(Session)[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()

    def test_worker_num_is_capped_at_entity_count(self, populated):
        populated.get(Session).map_async(_mp_worker.double_score, _worker_num=64,
                                         _calibrate=False)
        assert len(populated.get(Session)['doubled']) == 6


class TestGpuDeviceAssignment:
    """Item 3: requesting the GPU must always pass a device to the function."""

    def test_explicit_device_count_still_assigns_a_device(self, populated):
        """Regression: passing _gpu_max_device_num used to skip the branch that
        enables device assignment, silently running everything without a device."""
        populated.get(Session).map_async(_mp_worker.record_device, _worker_num=2,
                                         _use_gpu=True, _gpu_max_device_num=0)

        # No CUDA devices available -> every worker is pinned to 'cpu'
        assert set(populated.get(Session)['device']) == {'cpu'}

    def test_device_is_pinned_per_worker_not_per_entity(self, many):
        """Each worker must stick to one device, otherwise every process would
        initialise a CUDA context on every device. Device and pid are recorded in
        the same run, so they refer to the same pool."""
        many.get(Session).map_async(_mp_worker.record_device, _worker_num=2,
                                    _use_gpu=True, _gpu_max_device_num=2)

        state = many.get(Session)[['device', 'pid']]

        assert set(state['device']) <= {'cuda:0', 'cuda:1'}
        # Every worker used exactly one device, and no two workers shared one
        assert (state.groupby('pid')['device'].nunique() == 1).all()
        assert state['device'].nunique() == state['pid'].nunique()

    def test_gpu_run_skips_calibration(self, populated, capsys):
        populated.get(Session).map_async(_mp_worker.record_device, _worker_num=2,
                                         _use_gpu=True, _gpu_max_device_num=0)
        assert 'Start pool' in capsys.readouterr().out


class TestCalibration:
    """Item 4: cheap work should not pay for a process pool."""

    def test_cheap_work_runs_in_process(self, many, capsys):
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=4)

        output = capsys.readouterr().out
        assert 'running in this process' in output
        assert 'Start pool' not in output

        out = many.get(Session)[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()

    def test_calibration_can_be_disabled(self, many, capsys):
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2,
                                    _calibrate=False)

        assert 'Start pool' in capsys.readouterr().out

    def test_single_worker_never_starts_a_pool(self, many, capsys):
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=1)

        assert 'Start pool' not in capsys.readouterr().out
        out = many.get(Session)[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()

    def test_expensive_work_still_uses_the_pool(self, many, capsys, monkeypatch):
        """With a high per-entity cost the estimate must favour the pool."""
        from entarchy.core import entity as entity_module
        monkeypatch.setattr(entity_module, 'POOL_STARTUP_ESTIMATE_SECONDS', 0.0)

        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2)

        assert 'Start pool' in capsys.readouterr().out
