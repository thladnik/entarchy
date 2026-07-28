import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
import pytest

from conftest import Session
from entarchy.backend.sqlite import Serializer
from entarchy.tools import repair_blobs


def corrupt_row(db_path: str, entity_uuid: str, attr_name: str, payload: np.ndarray, ent_path: str,
                max_blob_size: int):
    """Recreate the legacy corruption: a pickled insert-row Series in value_blob."""

    serializer = Serializer()
    serializer.serialize(payload, ent_path, entity_uuid, attr_name, max_blob_size)

    legacy_row = pd.Series({
        attr_name: payload,
        'entity_uuid': entity_uuid,
        'name': attr_name,
        'data_type': 'blob',
        'analysis_uuid': None,
        '__serializer': serializer,
    })

    con = sqlite3.connect(db_path)
    con.execute(
        "UPDATE attributes SET value_blob = ? WHERE entity_uuid = ? AND name = ?",
        (pickle.dumps(legacy_row), entity_uuid, attr_name))
    con.commit()
    con.close()


class TestRepairTool:

    def test_repairs_legacy_corruption(self, populated):
        entity = populated.get(Session)[0]
        payload = np.arange(12.0)
        entity['victim'] = payload  # writes a healthy blob row

        db_path = os.path.join(populated.path, 'test.db')
        corrupt_row(db_path, entity.uuid, 'victim', payload, populated.path,
                    populated.max_blob_size)

        # Sanity: the corrupted row is now unreadable
        entity._attribute_cache.clear()
        with pytest.raises(Exception):
            _ = entity['victim']

        # Dry run must not change anything
        url = f'sqlite:///{db_path}'
        summary = repair_blobs.repair(url, apply_changes=False)
        assert summary['repaired'] == 1
        entity._attribute_cache.clear()
        with pytest.raises(Exception):
            _ = entity['victim']

        # Applying the repair restores the payload
        summary = repair_blobs.repair(url, apply_changes=True)
        assert summary['repaired'] == 1
        assert summary['unreadable'] == []

        entity._attribute_cache.clear()
        assert np.array_equal(entity['victim'], payload)

    def test_healthy_rows_untouched(self, populated):
        entity = populated.get(Session)[0]
        entity['fine'] = np.ones(5)

        db_path = os.path.join(populated.path, 'test.db')
        summary = repair_blobs.repair(f'sqlite:///{db_path}', apply_changes=True)
        assert summary['repaired'] == 0
        assert summary['healthy'] >= 1

        entity._attribute_cache.clear()
        assert np.array_equal(entity['fine'], np.ones(5))
