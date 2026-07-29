"""Exported ASDF archives must behave like the entarchy they came from.

The point of the archive format is that analysis and figure code keeps working
against it unchanged, so most of these tests assert equality between a live
entarchy and its archive rather than checking the archive in isolation.
"""
import os
import pickle

import numpy as np
import pytest

import entarchy
from entarchy.backend import SQLiteBackend

asdf = pytest.importorskip('asdf')

from entarchy.backend import asdf_store  # noqa: E402
from entarchy.backend.archive import BLOCK_DIR, INDEX_NAME, META_NAME, ArchiveReadOnlyError  # noqa: E402
from entarchy.tools import archive as archive_tool  # noqa: E402

from conftest import Animal, DeepArchy, Layer, Recording, Roi  # noqa: E402


class CustomThing:
    """Only exists so the exporter has something it cannot encode natively."""

    def __init__(self, payload):
        self.payload = payload

    def __eq__(self, other):
        return isinstance(other, CustomThing) and other.payload == self.payload


def _ragged(rng, count):
    return [rng.integers(0, 20, size=(int(rng.integers(2, 8)), 2)) for _ in range(count)]


@pytest.fixture()
def source(tmp_path):
    """A small entarchy holding one of every value shape entarchy supports."""
    base = (tmp_path / 'source').as_posix()
    ent = DeepArchy.create(base, SQLiteBackend(base, dbname='source.db'))
    rng = np.random.default_rng(0)

    with ent:
        animal = Animal(ent, _id='animal_1', _parent=ent.root)
        ent.add_new_entity(animal)
        animal['strain'] = 'wildtype'
        animal['age'] = 12
        animal['weight'] = 4.25
        animal['healthy'] = True
        animal['note'] = None

        for r in range(2):
            recording = Recording(ent, _id=f'rec_{r}', _parent=animal)
            ent.add_new_entity(recording)
            recording['rate'] = 10.0 + r
            recording['times'] = np.linspace(0, 1, 50)

            for l in range(2):
                layer = Layer(ent, _id=f'plane{l}', _parent=recording)
                ent.add_new_entity(layer)
                layer['depth'] = float(l * 15)
                layer['motion'] = rng.normal(size=(20, 3))

                for i in range(3):
                    roi = Roi(ent, _id=f'roi_{i}', _parent=layer)
                    ent.add_new_entity(roi)
                    roi['index'] = i
                    roi['good'] = (i != 1)
                    roi['dff'] = rng.normal(size=100).astype(np.float32)
                    # The shapes that motivated the ragged packing
                    roi['cluster_full_indices'] = _ragged(rng, 4)
                    roi['bs_cluster_full_indices'] = [_ragged(rng, int(rng.integers(1, 4)))
                                                      for _ in range(5)]
                    roi['cluster_unique_patch_indices'] = [(1, 2), (3, 4, 5)]
                    roi['raw_bytes'] = b'\x00\x01\x02binary'
                    roi['params'] = {'window': 120, 'percentile': 10.5, 'method': 'rolling'}

    # Special float values, which are stored as flags rather than values
    with ent:
        first = ent.get(Roi, 'index == 0')[0]
        first['odd_float'] = float('nan')
        first['big_float'] = float('inf')
        first['small_float'] = float('-inf')

    yield ent
    ent.backend.close()


@pytest.fixture()
def archived(source, tmp_path):
    """The source, exported and reopened as an archive."""
    destination = (tmp_path / 'archive').as_posix()
    stats = archive_tool.export(source, destination, verbose=False)

    ent = DeepArchy(destination)
    yield ent, stats
    ent.backend.close()
    asdf_store.close_asdf_files()


def _ids(collection):
    return sorted(e.id for e in collection)


class TestLayout:

    def test_expected_files_exist(self, archived, tmp_path):
        _, stats = archived
        destination = (tmp_path / 'archive').as_posix()

        assert os.path.exists(os.path.join(destination, 'entarchy.yaml'))
        assert os.path.exists(os.path.join(destination, INDEX_NAME))
        assert os.path.exists(os.path.join(destination, META_NAME))
        assert stats['block_files'] > 0

    def test_blocks_are_grouped_by_parent(self, archived, tmp_path):
        """One file per parent group, not one per entity."""
        destination = (tmp_path / 'archive').as_posix()
        block_files = os.listdir(os.path.join(destination, BLOCK_DIR))

        # 4 layers hold rois, 1 animal holds recordings, 2 recordings hold layers,
        #  the root holds the animal - far fewer files than the 17 entities
        assert 0 < len(block_files) <= 8

    def test_config_names_archive_backend(self, archived):
        ent, _ = archived
        assert type(ent.backend).__name__ == 'ArchiveBackend'


class TestQueryEquivalence:
    """The archive must answer queries exactly as the live entarchy does."""

    def test_plain_type_query(self, source, archived):
        ent, _ = archived
        assert _ids(ent.get(Roi)) == _ids(source.get(Roi))
        assert len(ent.get(Roi)) == 12

    def test_scalar_filter(self, source, archived):
        ent, _ = archived
        assert _ids(ent.get(Roi, 'index == 0')) == _ids(source.get(Roi, 'index == 0'))

    def test_boolean_filter(self, source, archived):
        ent, _ = archived
        assert _ids(ent.get(Roi, 'good == True')) == _ids(source.get(Roi, 'good == True'))

    def test_parent_attribute_filter(self, source, archived):
        ent, _ = archived
        query = '[Animal]strain == "wildtype"'
        assert _ids(ent.get(Roi, query)) == _ids(source.get(Roi, query))
        assert len(ent.get(Roi, query)) == 12

    def test_relative_parent_filter(self, source, archived):
        ent, _ = archived
        query = '../depth == 15.0'
        assert _ids(ent.get(Roi, query)) == _ids(source.get(Roi, query))

    def test_exist_filter(self, source, archived):
        ent, _ = archived
        query = 'EXIST(odd_float)'
        assert _ids(ent.get(Roi, query)) == _ids(source.get(Roi, query))

    def test_in_filter(self, source, archived):
        ent, _ = archived
        query = 'index IN (0, 2)'
        assert _ids(ent.get(Roi, query)) == _ids(source.get(Roi, query))

    def test_dataframe_matches(self, source, archived):
        ent, _ = archived
        columns = ['index', 'good']
        expected = source.get(Roi).dataframe_of(columns).sort_index()
        actual = ent.get(Roi).dataframe_of(columns).sort_index()

        assert list(actual.columns) == list(expected.columns)
        assert actual['index'].tolist() == expected['index'].tolist()
        assert actual['good'].tolist() == expected['good'].tolist()

    def test_hierarchy_traversal(self, source, archived):
        ent, _ = archived
        roi = ent.get(Roi, 'index == 0')[0]

        assert roi.parent.id.startswith('plane')
        assert roi.parent.parent.id.startswith('rec_')
        assert roi.parent.parent.parent.id == 'animal_1'
        assert roi.parent.parent.parent['strain'] == 'wildtype'


class TestValueFidelity:

    def test_scalars(self, source, archived):
        ent, _ = archived
        animal = ent.get(Animal)[0]
        original = source.get(Animal)[0]

        assert animal['strain'] == original['strain'] == 'wildtype'
        assert animal['age'] == original['age'] == 12
        assert animal['weight'] == original['weight'] == 4.25
        assert animal['healthy'] is True

    def test_special_floats(self, source, archived):
        ent, _ = archived
        roi = ent.get(Roi, 'EXIST(odd_float)')[0]

        assert np.isnan(roi['odd_float'])
        assert roi['big_float'] == float('inf')
        assert roi['small_float'] == float('-inf')

    def test_arrays_bit_identical(self, source, archived):
        ent, _ = archived
        for original, restored in zip(sorted(source.get(Layer), key=lambda e: e.uuid),
                                      sorted(ent.get(Layer), key=lambda e: e.uuid)):
            np.testing.assert_array_equal(restored['motion'], original['motion'])
            assert restored['motion'].dtype == original['motion'].dtype

    def test_array_dtype_preserved(self, source, archived):
        ent, _ = archived
        roi = ent.get(Roi, 'index == 0')[0]
        assert roi['dff'].dtype == np.float32

    def test_arrays_are_real_ndarrays(self, archived):
        """asdf hands back proxy objects; code that dispatches on type must still work."""
        ent, _ = archived
        roi = ent.get(Roi, 'index == 0')[0]

        assert type(roi['dff']) is np.ndarray
        assert all(type(item) is np.ndarray for item in roi['cluster_full_indices'])

        # Pickling is what map_async and the import path both depend on
        assert pickle.loads(pickle.dumps(roi['dff'])).shape == (100,)

    def test_bytes_stay_bytes(self, source, archived):
        ent, _ = archived
        roi = ent.get(Roi, 'index == 0')[0]

        assert isinstance(roi['raw_bytes'], bytes)
        assert roi['raw_bytes'] == b'\x00\x01\x02binary'

    def test_dict_attribute(self, source, archived):
        ent, _ = archived
        roi = ent.get(Roi, 'index == 0')[0]

        assert roi['params'] == {'window': 120, 'percentile': 10.5, 'method': 'rolling'}

    def test_ragged_array_list(self, source, archived):
        ent, _ = archived
        for original, restored in zip(sorted(source.get(Roi), key=lambda e: e.uuid),
                                      sorted(ent.get(Roi), key=lambda e: e.uuid)):
            expected = original['cluster_full_indices']
            actual = restored['cluster_full_indices']

            assert isinstance(actual, list)
            assert len(actual) == len(expected)
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(right, left)

    def test_nested_ragged_array_list(self, source, archived):
        ent, _ = archived
        roi_source = sorted(source.get(Roi), key=lambda e: e.uuid)[0]
        roi_archive = sorted(ent.get(Roi), key=lambda e: e.uuid)[0]

        expected = roi_source['bs_cluster_full_indices']
        actual = roi_archive['bs_cluster_full_indices']

        assert len(actual) == len(expected)
        for outer_expected, outer_actual in zip(expected, actual):
            assert len(outer_actual) == len(outer_expected)
            for left, right in zip(outer_expected, outer_actual):
                np.testing.assert_array_equal(right, left)

    def test_tuples_survive(self, source, archived):
        """YAML has no tuple, so this is the case a naive encoding loses."""
        ent, _ = archived
        roi = ent.get(Roi, 'index == 0')[0]
        value = roi['cluster_unique_patch_indices']

        assert value == [(1, 2), (3, 4, 5)]
        assert all(isinstance(item, tuple) for item in value)


class TestRaggedPacking:

    def test_packing_keeps_block_count_low(self, tmp_path):
        """A list of arrays must not become one binary block per array."""
        rng = np.random.default_rng(1)
        value = [[rng.integers(0, 10, size=(5, 2)) for _ in range(3)] for _ in range(200)]

        encoded = asdf_store.encode(value)
        path = (tmp_path / 'packed.asdf').as_posix()
        handle = asdf.AsdfFile({'blobs': {'x': encoded}})
        handle.write_to(path)
        handle.close()

        with asdf.open(path) as f:
            block_count = len(f._blocks.blocks)

        # 600 arrays, stored as one data block plus two offset blocks
        assert block_count == 3

        restored = asdf_store.decode(asdf.open(path)['blobs']['x'])
        assert len(restored) == 200
        for outer_expected, outer_actual in zip(value, restored):
            for left, right in zip(outer_expected, outer_actual):
                np.testing.assert_array_equal(right, left)

    def test_mixed_shapes_still_round_trip(self):
        """Arrays that cannot share a block fall back to a plain list."""
        value = [np.zeros((2, 2)), np.zeros((3, 5))]

        restored = asdf_store.decode(asdf_store.encode(value))

        assert len(restored) == 2
        assert restored[0].shape == (2, 2)
        assert restored[1].shape == (3, 5)

    def test_empty_list(self):
        assert asdf_store.decode(asdf_store.encode([])) == []


class TestOpenFileCache:
    """Block files are cached per process; eviction must stay correct and rare."""

    def test_default_limit_exceeds_typical_group_count(self):
        # A limit below the number of groups a read pass touches makes every read
        #  evict a file the next one wants; measured 13x slower over 12 groups
        assert asdf_store._OPEN_FILE_LIMIT >= 32

    def test_values_survive_eviction(self, source, archived):
        """A tiny cache must give the same answers, only more slowly."""
        ent, _ = archived
        asdf_store.close_asdf_files()
        original_limit = asdf_store._OPEN_FILE_LIMIT

        try:
            asdf_store.set_open_file_limit(1)

            for original, restored in zip(sorted(source.get(Roi), key=lambda e: e.uuid),
                                          sorted(ent.get(Roi), key=lambda e: e.uuid)):
                np.testing.assert_array_equal(restored['dff'], original['dff'])
        finally:
            asdf_store.set_open_file_limit(original_limit)

    def test_limit_is_configurable_from_config(self, archived, tmp_path):
        from entarchy.backend.archive import ArchiveBackend

        destination = (tmp_path / 'archive').as_posix()
        original_limit = asdf_store._OPEN_FILE_LIMIT

        try:
            ArchiveBackend(destination, open_file_limit=17)
            assert asdf_store._OPEN_FILE_LIMIT == 17
        finally:
            asdf_store.set_open_file_limit(original_limit)

    def test_rejects_nonsense_limit(self):
        with pytest.raises(ValueError):
            asdf_store.set_open_file_limit(0)


class TestPortabilityReport:

    def test_pickle_fallback_is_reported(self, tmp_path):
        report = asdf_store.EncodingReport()
        asdf_store.encode(CustomThing([1, 2, 3]), report, 'roi_0.thing')

        assert not report.is_fully_portable
        assert report.pickled == [('roi_0.thing', 'CustomThing')]
        assert 'CustomThing' in report.summary()

    def test_pickle_fallback_still_round_trips(self):
        value = CustomThing([1, 2, 3])
        assert asdf_store.decode(asdf_store.encode(value)) == value

    def test_native_values_report_clean(self):
        report = asdf_store.EncodingReport()
        asdf_store.encode({'a': [1, 2], 'b': np.zeros(3)}, report)

        assert report.is_fully_portable
        assert report.summary() == 'all values encoded natively'

    def test_export_reports_unencodable_values(self, source, tmp_path, capsys):
        with source:
            roi = source.get(Roi, 'index == 0')[0]
            roi['custom'] = CustomThing([9])

        destination = (tmp_path / 'archive_custom').as_posix()
        stats = archive_tool.export(source, destination, verbose=True)

        assert not stats['report'].is_fully_portable
        assert 'CustomThing' in capsys.readouterr().out


class TestReadOnly:

    def test_setting_attribute_raises(self, archived):
        ent, _ = archived
        roi = ent.get(Roi, 'index == 0')[0]

        with pytest.raises(ArchiveReadOnlyError, match='read-only'):
            with ent:
                roi['new_attribute'] = 1

    def test_adding_entity_raises(self, archived):
        ent, _ = archived
        animal = ent.get(Animal)[0]

        with pytest.raises(ArchiveReadOnlyError):
            with ent:
                recording = Recording(ent, _id='rec_new', _parent=animal)
                ent.add_new_entity(recording)

    def test_delete_raises(self, archived):
        ent, _ = archived
        with pytest.raises(ArchiveReadOnlyError):
            ent.backend.delete(confirm=True)

    def test_error_points_at_import(self, archived):
        ent, _ = archived
        with pytest.raises(ArchiveReadOnlyError, match='tools.archive'):
            ent.backend.add_entities([])


class TestRebuildIndex:
    """index.sqlite is a cache; meta.asdf is the source of truth."""

    def test_rebuild_reproduces_queries(self, source, archived, tmp_path):
        ent, _ = archived
        destination = (tmp_path / 'archive').as_posix()
        expected = _ids(ent.get(Roi, 'good == True'))

        ent.backend.close()
        os.remove(os.path.join(destination, INDEX_NAME))

        archive_tool.rebuild_index(destination, verbose=False)

        rebuilt = DeepArchy(destination)
        try:
            assert _ids(rebuilt.get(Roi, 'good == True')) == expected
            assert _ids(rebuilt.get(Roi)) == _ids(source.get(Roi))
        finally:
            rebuilt.backend.close()

    def test_rebuild_restores_blob_access(self, source, archived, tmp_path):
        ent, _ = archived
        destination = (tmp_path / 'archive').as_posix()
        ent.backend.close()
        asdf_store.close_asdf_files()
        os.remove(os.path.join(destination, INDEX_NAME))

        archive_tool.rebuild_index(destination, verbose=False)

        rebuilt = DeepArchy(destination)
        try:
            original = sorted(source.get(Layer), key=lambda e: e.uuid)[0]
            restored = sorted(rebuilt.get(Layer), key=lambda e: e.uuid)[0]
            np.testing.assert_array_equal(restored['motion'], original['motion'])
        finally:
            rebuilt.backend.close()

    def test_missing_index_explains_rebuild(self, archived, tmp_path):
        ent, _ = archived
        destination = (tmp_path / 'archive').as_posix()
        ent.backend.close()
        os.remove(os.path.join(destination, INDEX_NAME))

        broken = DeepArchy(destination)
        with pytest.raises(FileNotFoundError, match='rebuild'):
            # Collections are lazy, so the index is only opened on evaluation
            len(broken.get(Roi))


class TestImportBack:

    def test_imported_entarchy_is_writable(self, source, archived, tmp_path):
        destination = (tmp_path / 'archive').as_posix()
        ent, _ = archived
        ent.backend.close()
        asdf_store.close_asdf_files()

        imported_path = (tmp_path / 'imported').as_posix()
        archive_tool.import_archive(destination, imported_path, verbose=False)

        imported = DeepArchy(imported_path)
        try:
            assert _ids(imported.get(Roi)) == _ids(source.get(Roi))

            with imported:
                roi = imported.get(Roi, 'index == 0')[0]
                roi['added_later'] = 42

            assert imported.get(Roi, 'added_later == 42')[0]['added_later'] == 42
        finally:
            imported.backend.close()

    def test_imported_blobs_match(self, source, archived, tmp_path):
        destination = (tmp_path / 'archive').as_posix()
        ent, _ = archived
        ent.backend.close()
        asdf_store.close_asdf_files()

        imported_path = (tmp_path / 'imported').as_posix()
        archive_tool.import_archive(destination, imported_path, verbose=False)

        imported = DeepArchy(imported_path)
        try:
            original = sorted(source.get(Roi), key=lambda e: e.uuid)[0]
            restored = sorted(imported.get(Roi), key=lambda e: e.uuid)[0]

            np.testing.assert_array_equal(restored['dff'], original['dff'])
            assert restored['raw_bytes'] == original['raw_bytes']
            for left, right in zip(original['cluster_full_indices'],
                                   restored['cluster_full_indices']):
                np.testing.assert_array_equal(right, left)
        finally:
            imported.backend.close()

    def test_imported_entarchy_has_no_asdf_pointers(self, archived, tmp_path):
        destination = (tmp_path / 'archive').as_posix()
        ent, _ = archived
        ent.backend.close()
        asdf_store.close_asdf_files()

        imported_path = (tmp_path / 'imported').as_posix()
        archive_tool.import_archive(destination, imported_path, verbose=False)

        imported = DeepArchy(imported_path)
        try:
            from sqlalchemy.orm import Session

            from entarchy.backend.sql import AttributeTable

            with Session(imported.backend.sql_engine) as session:
                rows = session.query(AttributeTable).filter(
                    AttributeTable.data_type == 'blob').all()
                stores = [pickle.loads(row.value_blob)._store for row in rows]

            assert len(stores) > 0
            assert not any(store.startswith(asdf_store.STORE_PREFIX) for store in stores)
        finally:
            imported.backend.close()


@pytest.mark.slow
class TestParallelReads:
    """Several worker processes reading one archive at once."""

    def test_map_async_reads_archive(self, archived):
        import _mp_worker

        ent, _ = archived
        # Raises RuntimeError if any entity failed, so completing is the assertion
        ent.get(Roi).map_async(_mp_worker.check_archive_roi, expected_length=100,
                               _worker_num=2, _calibrate=False)

    def test_map_async_write_is_refused(self, archived):
        import _mp_worker

        ent, _ = archived
        with pytest.raises(RuntimeError, match='read-only'):
            ent.get(Roi).map_async(_mp_worker.write_to_archive_roi,
                                   _worker_num=2, _calibrate=False)


class TestSubsetExport:

    def test_collection_export_includes_ancestors(self, source, tmp_path):
        """Exporting a subset must keep parents, or parent lookups break."""
        destination = (tmp_path / 'subset').as_posix()
        collection = source.get(Roi, 'index == 0')
        archive_tool.export(source, destination, collection=collection, verbose=False)

        ent = DeepArchy(destination)
        try:
            assert len(ent.get(Roi)) == 4
            assert len(ent.get(Animal)) == 1

            roi = ent.get(Roi)[0]
            assert roi.parent.parent.parent['strain'] == 'wildtype'
            assert len(ent.get(Roi, '[Animal]strain == "wildtype"')) == 4
        finally:
            ent.backend.close()
            asdf_store.close_asdf_files()


class TestCommandLine:
    """The CLI must work without the schema package being importable."""

    def test_export_rebuild_import_roundtrip(self, source, tmp_path, capsys):
        destination = (tmp_path / 'cli_archive').as_posix()
        imported = (tmp_path / 'cli_imported').as_posix()
        source.backend.close()

        assert archive_tool.main(['export', source.path, destination]) == 0
        assert os.path.exists(os.path.join(destination, META_NAME))

        os.remove(os.path.join(destination, INDEX_NAME))
        assert archive_tool.main(['rebuild', destination]) == 0
        assert os.path.exists(os.path.join(destination, INDEX_NAME))

        assert archive_tool.main(['import', destination, imported]) == 0
        assert os.path.exists(os.path.join(imported, 'entarchy.db'))

        ent = DeepArchy(imported)
        try:
            assert len(ent.get(Roi)) == 12
        finally:
            ent.backend.close()
            asdf_store.close_asdf_files()

    def test_subset_needs_entarchy_class(self, source, tmp_path):
        destination = (tmp_path / 'cli_subset').as_posix()
        source.backend.close()

        with pytest.raises(SystemExit, match='entarchy-class'):
            archive_tool.main(['export', source.path, destination, '--type', 'Roi'])

    def test_subset_with_entarchy_class(self, source, tmp_path):
        destination = (tmp_path / 'cli_subset_ok').as_posix()
        source_path = source.path
        source.backend.close()

        archive_tool.main(['export', source_path, destination,
                           '--entarchy-class', 'conftest.DeepArchy',
                           '--type', 'Roi', '--query', 'index == 0'])

        ent = DeepArchy(destination)
        try:
            assert len(ent.get(Roi)) == 4
        finally:
            ent.backend.close()
            asdf_store.close_asdf_files()


class TestExportGuards:

    def test_refuses_existing_destination(self, source, tmp_path):
        destination = (tmp_path / 'twice').as_posix()
        archive_tool.export(source, destination, verbose=False)

        with pytest.raises(archive_tool.ExportError, match='already exists'):
            archive_tool.export(source, destination, verbose=False)

    def test_overwrite_replaces(self, source, tmp_path):
        destination = (tmp_path / 'twice').as_posix()
        archive_tool.export(source, destination, verbose=False)
        stats = archive_tool.export(source, destination, overwrite=True, verbose=False)

        assert stats['entities'] > 0

    def test_entarchy_to_asdf(self, source, tmp_path):
        destination = (tmp_path / 'convenience').as_posix()
        stats = source.to_asdf(destination, verbose=False)

        assert stats['entities'] > 0
        assert os.path.exists(os.path.join(destination, META_NAME))

    def test_collection_to_asdf(self, source, tmp_path):
        destination = (tmp_path / 'convenience_collection').as_posix()
        stats = source.get(Roi, 'index == 0').to_asdf(destination, verbose=False)

        ent = DeepArchy(destination)
        try:
            assert len(ent.get(Roi)) == 4
            assert stats['entities'] > 4  # ancestors came along
        finally:
            ent.backend.close()
            asdf_store.close_asdf_files()

    def test_export_accepts_a_path(self, source, tmp_path):
        destination = (tmp_path / 'from_path').as_posix()
        source.backend.close()

        stats = archive_tool.export(source.path, destination, verbose=False)

        assert stats['entities'] > 0
