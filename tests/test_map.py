import multiprocessing

import pytest

import _mp_worker
from conftest import Session, Subject
from entarchy.core.entity import shutdown_worker_pool


@pytest.fixture(autouse=True)
def _release_pool():
    """The worker pool outlives a single call, so release it between tests."""
    yield
    shutdown_worker_pool()


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


@pytest.fixture()
def grouped(ent):
    """Sessions spread over three parents, to exercise locality grouping."""
    with ent:
        for s in range(3):
            subject = Subject(ent, _id=f'subject_{s}', _parent=ent.root)
            ent.add_new_entity(subject)
            subject['label'] = f'label_{s}'

            for i in range(6):
                session = Session(ent, _id=f'sess_{s}_{i}', _parent=subject)
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

    def test_work_runs_in_worker_processes(self, many):
        """Deterministic: every task ran in a child, and never in more processes
        than the pool has workers."""
        import os

        many.get(Session).map_async(_mp_worker.record_worker_state,
                                    _worker_num=2, _calibrate=False)

        pids = set(many.get(Session)['pid'])
        assert os.getpid() not in pids
        assert 1 <= len(pids) <= 2

    def test_work_is_spread_over_processes(self, many):
        """Which worker takes a chunk is a race, so hand out single tasks and hold
        each one briefly; otherwise one worker can drain the queue on its own."""
        many.get(Session).map_async(_mp_worker.record_worker_state, _worker_num=2,
                                    _calibrate=False, _chunk_size=1, sleep=0.05)

        assert len(set(many.get(Session)['pid'])) > 1

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
        assert 'worker pool' in capsys.readouterr().out


class TestCalibration:
    """Item 4: cheap work should not pay for a process pool."""

    def test_cheap_work_runs_in_process(self, many, capsys):
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=4)

        output = capsys.readouterr().out
        assert 'running in this process' in output
        assert 'worker pool' not in output

        out = many.get(Session)[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()

    def test_calibration_can_be_disabled(self, many, capsys):
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2,
                                    _calibrate=False)

        assert 'worker pool' in capsys.readouterr().out

    def test_single_worker_never_starts_a_pool(self, many, capsys):
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=1)

        assert 'worker pool' not in capsys.readouterr().out
        out = many.get(Session)[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()

    def test_expensive_work_still_uses_the_pool(self, many, capsys, monkeypatch):
        """With a high per-entity cost the estimate must favour the pool."""
        from entarchy.core import entity as entity_module
        monkeypatch.setattr(entity_module, 'POOL_STARTUP_ESTIMATE_SECONDS', 0.0)

        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2)

        assert 'worker pool' in capsys.readouterr().out


class TestPoolReuse:
    """Item 1: the pool survives between calls, so startup is paid once."""

    def test_second_call_reuses_the_pool(self, many, capsys):
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2, _calibrate=False)
        assert 'Start worker pool' in capsys.readouterr().out

        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2, _calibrate=False)
        assert 'Reuse worker pool' in capsys.readouterr().out

    def test_same_workers_serve_both_calls(self, many):
        """No new processes appear for the second call. Which of the two workers
        picks up a given chunk is a race, so compare the union rather than the
        individual sets: a rebuilt pool would push the total above worker_num."""
        many.get(Session).map_async(_mp_worker.record_worker_state, _worker_num=2,
                                    _calibrate=False)
        first = set(many.get(Session)['pid'])

        many.get(Session).map_async(_mp_worker.record_worker_state, _worker_num=2,
                                    _calibrate=False)
        second = set(many.get(Session)['pid'])

        assert len(first | second) <= 2

    def test_warm_pool_lowers_the_startup_estimate(self, many, capsys, monkeypatch):
        """Calibration weighs a running pool by its dispatch cost, not by the cost
        of starting one from scratch."""
        from entarchy.core import entity as entity_module
        monkeypatch.setattr(entity_module, 'POOL_STARTUP_ESTIMATE_SECONDS', 1e6)
        monkeypatch.setattr(entity_module, 'POOL_WARM_STARTUP_SECONDS', 0.0)

        # Cold: starting a pool is ruled out
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2)
        assert 'running in this process' in capsys.readouterr().out

        # Warm the pool
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2, _calibrate=False)
        capsys.readouterr()

        # Warm: the same work is now worth dispatching
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2)
        assert 'Reuse worker pool' in capsys.readouterr().out

    def test_changing_worker_count_rebuilds_the_pool(self, many, capsys):
        many.get(Session).map_async(_mp_worker.record_worker_state, _worker_num=2,
                                    _calibrate=False)
        first = set(many.get(Session)['pid'])
        capsys.readouterr()

        many.get(Session).map_async(_mp_worker.record_worker_state, _worker_num=3,
                                    _calibrate=False)
        assert 'Start worker pool' in capsys.readouterr().out
        assert set(many.get(Session)['pid']).isdisjoint(first)

    def test_shutdown_releases_workers(self, many, capsys):
        many.get(Session).map_async(_mp_worker.record_worker_state, _worker_num=2,
                                    _calibrate=False)
        first = set(many.get(Session)['pid'])

        shutdown_worker_pool()
        capsys.readouterr()

        many.get(Session).map_async(_mp_worker.record_worker_state, _worker_num=2,
                                    _calibrate=False)
        assert 'Start worker pool' in capsys.readouterr().out
        assert set(many.get(Session)['pid']).isdisjoint(first)

    def test_shutdown_is_safe_without_a_pool(self):
        shutdown_worker_pool()
        shutdown_worker_pool()


class TestLocality:
    """Item 1: entities that share a parent are processed together."""

    def test_all_entities_still_processed(self, grouped):
        grouped.get(Session).map_async(_mp_worker.record_locality, _worker_num=2,
                                       _calibrate=False)

        state = grouped.get(Session)[['parent_id', 'parent_label']]
        assert len(state) == 18
        assert set(state['parent_id']) == {'subject_0', 'subject_1', 'subject_2'}

    def test_each_worker_sees_parents_in_contiguous_runs(self, grouped):
        grouped.get(Session).map_async(_mp_worker.record_locality, _worker_num=2,
                                       _calibrate=False)

        state = grouped.get(Session)[['parent_id', 'pid', 'order']]

        for pid, block in state.groupby('pid'):
            sequence = list(block.sort_values('order')['parent_id'])
            transitions = sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)
            # A worker touching k parents should switch k-1 times, not bounce
            assert transitions == len(set(sequence)) - 1

    def test_parents_are_released_when_the_group_changes(self, grouped):
        """Only the current entity and its ancestors stay registered."""
        grouped.get(Session).map_async(_mp_worker.record_locality, _worker_num=2,
                                       _calibrate=False)

        # entity + its parent (+ the parent's own parent once touched)
        assert max(grouped.get(Session)['registry_size']) <= 3

    def test_locality_can_be_disabled(self, grouped):
        grouped.get(Session).map_async(_mp_worker.record_locality, _worker_num=2,
                                       _calibrate=False, _locality=False)
        assert len(grouped.get(Session)['parent_id']) == 18

    def test_chunks_are_sized_to_the_groups(self, grouped, capsys):
        """Chunks that straddle group boundaries make workers release a parent and
        load it again, so the default chunk size follows the group size."""
        grouped.get(Session).map_async(_mp_worker.record_locality, _worker_num=2,
                                       _calibrate=False)
        assert 'chunk size 6' in capsys.readouterr().out  # 3 groups of 6 sessions

    def test_chunk_size_is_capped_for_load_balance(self, ent, capsys):
        """A single huge group must still spread over the workers."""
        with ent:
            subject = Subject(ent, _id='only', _parent=ent.root)
            ent.add_new_entity(subject)
            subject['label'] = 'x'
            for i in range(20):
                session = Session(ent, _id=f's{i}', _parent=subject)
                ent.add_new_entity(session)
                session['index'] = i

        ent.get(Session).map_async(_mp_worker.record_locality, _worker_num=2,
                                   _calibrate=False)

        # 20 entities in one group, but capped at entity_count // worker_num
        assert 'chunk size 10' in capsys.readouterr().out


class TestErrorCollection:
    """Item 1: one bad entity must not abandon the rest of the collection."""

    def test_remaining_entities_are_still_processed(self, many):
        with pytest.raises(RuntimeError, match='failed on'):
            many.get(Session).map_async(_mp_worker.fail_on_odd_index, _worker_num=2,
                                        _calibrate=False)

        # The six even-indexed entities completed and were committed
        succeeded = many.get(Session, 'EXIST(succeeded)')
        assert sorted(succeeded['index']) == [0, 2, 4, 6, 8, 10]

    def test_summary_reports_the_failures(self, many, capsys):
        with pytest.raises(RuntimeError):
            many.get(Session).map_async(_mp_worker.fail_on_odd_index, _worker_num=2,
                                        _calibrate=False)

        output = capsys.readouterr().out
        assert '6 of 12 entities failed' in output

    def test_error_carries_the_original_traceback(self, many):
        with pytest.raises(RuntimeError, match='deliberate failure'):
            many.get(Session).map_async(_mp_worker.fail_on_odd_index, _worker_num=2,
                                        _calibrate=False)

    def test_failures_in_the_in_process_path_are_collected(self, many):
        with pytest.raises(RuntimeError, match='failed on 6 of 12'):
            many.get(Session).map_async(_mp_worker.fail_on_odd_index, _worker_num=1)

    def test_success_does_not_raise(self, many):
        many.get(Session).map_async(_mp_worker.double_score, _worker_num=2,
                                    _calibrate=False)
        assert len(many.get(Session)['doubled']) == 12
