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
    _implementation_version = '0.1'
    _implementation_compat_version_list = ['0.1']
    _hierarchy_root_type = Subject


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
    _implementation_version = '0.1'
    _implementation_compat_version_list = ['0.1']
    _hierarchy_root_type = Animal


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
        animal = Animal(e, _id='animal_1', _parent=e.root)
        e.add_new_entity(animal)
        animal['strain'] = 'wildtype'
        animal['age'] = 12

        for r in range(2):
            recording = Recording(e, _id=f'rec_{r}', _parent=animal)
            e.add_new_entity(recording)
            recording['rate'] = 10.0 + r

            for l in range(2):
                layer = Layer(e, _id=f'plane{l}', _parent=recording)
                e.add_new_entity(layer)
                layer['depth'] = float(l * 15)

                for i in range(3):
                    roi = Roi(e, _id=f'roi_{i}', _parent=layer)
                    e.add_new_entity(roi)
                    roi['index'] = i

    yield e
    e.backend.close()


@pytest.fixture()
def populated(ent):
    with ent:
        subject = Subject(ent, _id='subject_a', _parent=ent.root)
        ent.add_new_entity(subject)
        subject['strain'] = 'wildtype'

        for i in range(6):
            sess = Session(ent, _id=f'sess_{i}', _parent=subject)
            ent.add_new_entity(sess)
            sess['index'] = i
            sess['score'] = float(i) * 1.5
            sess['flag'] = (i % 2 == 0)

    return ent
