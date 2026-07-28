"""Worker functions for map_async tests.

Kept in an importable module (not the test file), because spawned worker
processes must be able to import the function by qualified name.
"""
import os


def double_score(session):
    session['doubled'] = float(session['score']) * 2.0


def record_worker_state(session):
    """Record whether this worker already had an open backend when the task started.

    The first task handled by a worker finds no connection; every later task in the
    same worker must reuse the one opened by its predecessor.
    """
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


def fail_on_odd_index(session):
    """Fail for half the entities, to exercise per-entity error collection."""
    if session['index'] % 2 == 1:
        raise ValueError(f'deliberate failure for {session.id}')
    session['succeeded'] = True
