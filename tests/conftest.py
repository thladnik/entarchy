import pathlib
import sys

# Make the repository root importable without installation
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# Make this directory importable in spawned worker processes (map_async tests)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pytest

import entarchy
from entarchy.backend import SQLiteBackend


class Subject(entarchy.Entity):
    pass


class Session(entarchy.Entity):
    pass


Subject.add_child_entity_type(Session)


class LabArchy(entarchy.Entarchy):
    _entity_types = (Subject,)


# A deeper hierarchy (4 levels) for multi-level parent traversal tests:
#   Animal > Recording > Layer > Roi
class Animal(entarchy.Entity):
    pass


class Recording(entarchy.Entity):
    pass


class Layer(entarchy.Entity):
    pass


class Roi(entarchy.Entity):
    pass


Animal.add_child_entity_type(Recording)
Recording.add_child_entity_type(Layer)
Layer.add_child_entity_type(Roi)


class DeepArchy(entarchy.Entarchy):
    _entity_types = (Animal,)


@pytest.fixture()
def ent(tmp_path):
    base = (tmp_path / 'archy').as_posix()
    # Deliberately use the instance returned by create() - it must be fully usable
    e = LabArchy.create(base, SQLiteBackend(base, dbname='test.db'))
    yield e
    e.backend.close()


@pytest.fixture()
def deep(tmp_path):
    """Entarchy with a 4-level hierarchy: 1 animal > 2 recordings > 2 layers > 3 rois."""
    base = (tmp_path / 'deep').as_posix()
    e = DeepArchy.create(base, SQLiteBackend(base, dbname='deep.db'))

    with e:
        animal = e.add_child(Animal, 'animal_1')
        animal['strain'] = 'wildtype'
        animal['age'] = 12

        for r in range(2):
            recording = animal.add_child(Recording, f'rec_{r}')
            recording['rate'] = 10.0 + r

            for l in range(2):
                layer = recording.add_child(Layer, f'plane{l}')
                layer['depth'] = float(l * 15)

                for i in range(3):
                    roi = layer.add_child(Roi, f'roi_{i}')
                    roi['index'] = i

    yield e
    e.backend.close()


def make_link_row(ent, link_type, linker_uuid, linked_uuid, link_uuid=None):
    """Insert a link and its carrier entity directly, without the write API.

    The creation API comes with the link write path; until then tests exercise
    the schema through plain inserts.
    """
    import uuid as uuid_module

    from sqlalchemy.orm import Session as SASession

    from entarchy.backend.sql import EntityTable, EntityTypeTable, Link

    link_uuid = link_uuid or str(uuid_module.uuid4())

    with SASession(ent.backend.sql_engine) as session:
        link_entity_pk = session.query(EntityTypeTable).filter(
            EntityTypeTable.name == 'LinkEntity').one().pk

        session.add(EntityTable(uuid=link_uuid, id=f'{link_type}@{linked_uuid}',
                                parent_uuid=linker_uuid, entity_type_pk=link_entity_pk))
        session.add(Link(link_uuid=link_uuid, link_type=link_type,
                         linker_uuid=linker_uuid, linked_uuid=linked_uuid))
        session.commit()

    return link_uuid


@pytest.fixture()
def populated(ent):
    with ent:
        subject = ent.add_child(Subject, 'subject_a')
        subject['strain'] = 'wildtype'

        for i in range(6):
            sess = subject.add_child(Session, f'sess_{i}')
            sess['index'] = i
            sess['score'] = float(i) * 1.5
            sess['flag'] = (i % 2 == 0)

    return ent
