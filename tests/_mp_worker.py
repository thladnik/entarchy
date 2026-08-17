"""Worker functions for map_async tests.

Kept in an importable module (not the test file), because spawned worker
processes must be able to import the function by qualified name.
"""
import os


def double_score(session):
    session['doubled'] = float(session['score']) * 2.0


def record_worker_state(session, sleep: float = 0.0):
    """Record whether this worker already had an open backend when the task started.

    The first task handled by a worker finds no connection; every later task in the
    same worker must reuse the one opened by its predecessor.

    `sleep` holds the task briefly, so a single worker cannot drain the queue
    before its peers are scheduled.
    """
    if sleep:
        import time
        time.sleep(sleep)

    entarchy = session.entarchy

    # Read the raw attribute, not the property, so this check does not itself
    #  open a connection
    session['backend_was_open'] = entarchy._backend is not None
    session['pid'] = os.getpid()
    session['registry_size'] = len(entarchy._entities)


def record_device(session, _use_gpu_device=None):
    """Record the GPU device keyword the mapper passed in, and the worker that ran it."""
    session['device'] = str(_use_gpu_device)
    session['pid'] = os.getpid()


def record_locality(session):
    """Record which worker ran this entity, when, and which parent it belongs to."""
    import time

    # Touch a parent attribute, so the parent is materialised and cached
    session['parent_label'] = session.parent['label']
    session['parent_id'] = session.parent.id
    session['pid'] = os.getpid()
    session['order'] = time.perf_counter_ns()
    session['registry_size'] = len(session.entarchy._entities)


def check_archive_roi(roi, expected_length: int):
    """Read arrays out of an archive without writing anything back.

    Each worker opens the archive itself, so this covers several processes
    reading the same ASDF block files and index concurrently. map_async
    discards return values, so mismatches are raised rather than returned.
    """
    import numpy as np

    dff = roi['dff']
    if type(dff) is not np.ndarray:
        raise TypeError(f'expected ndarray from archive, got {type(dff).__name__}')
    if len(dff) != expected_length:
        raise ValueError(f'expected {expected_length} samples, got {len(dff)}')
    if len(roi['cluster_full_indices']) == 0:
        raise ValueError('ragged attribute came back empty')


def write_to_archive_roi(roi):
    """Writing to an archive must fail rather than silently do nothing."""
    roi['should_not_persist'] = 1


def scale_link_value(link, factor: float = 2.0):
    """Analysis over a link collection, reaching through to an endpoint."""
    link['scaled'] = float(link['mean_dff']) * factor
    link['linked_index'] = int(link.linked['index'])


def fail_on_odd_index(session):
    """Fail for half the entities, to exercise per-entity error collection."""
    if session['index'] % 2 == 1:
        raise ValueError(f'deliberate failure for {session.id}')
    session['succeeded'] = True


def record_order(entity):
    """Stamp when this entity was processed, to check processing order."""
    import time

    entity['processed_at'] = time.perf_counter_ns()
