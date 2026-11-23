"""The stored hierarchy is authoritative and grows: types are added to an open
entarchy, classes bind to them by name, and a type nobody bound still works."""
import os
import pickle

import pytest

import entarchy
from entarchy.backend import SQLiteBackend
from conftest import LabArchy, Session, Subject

import _mp_worker


def plain(tmp_path, name='plain', cls=entarchy.Entarchy):
    base = (tmp_path / name).as_posix()
    return cls.create(base, SQLiteBackend(base, dbname='plain.db'))


@pytest.fixture()
def bare(tmp_path):
    """A plain Entarchy with no types declared anywhere."""
    ent = plain(tmp_path)
    yield ent
    ent.backend.close()


class TestAPlainEntarchy:

    def test_starts_with_only_the_built_in_types(self, bare):
        assert bare.hierarchy == {'EntarchyEntity': {}, 'AnalysisEntity': {}, 'LinkEntity': {}}
        assert bare.entity_types == ()
        assert bare.root.child_entity_types() == ()

    def test_grows_by_name(self, bare):
        Subject_ = bare.add_entity_type('Subject')
        Session_ = bare.add_entity_type('Session', parent=Subject_)

        assert bare.hierarchy['Subject'] == {'Session': {}}
        assert issubclass(Subject_, entarchy.Entity) and Subject_.__name__ == 'Subject'
        assert bare.get_entity_type('Session') is Session_

        subject = bare.root.add_child(Subject_, 'subj_1', strain='wt')
        session = subject.add_child(Session_, 'sess_1', rate=10.0)

        assert len(bare.get('Subject')) == 1
        assert bare.get('Session')[0]['rate'] == 10.0
        assert session.parent.uuid == subject.uuid

    def test_add_child_entity_type_adds_a_root_type(self, bare):
        Subject_ = bare.add_child_entity_type('Subject')
        assert bare.child_type_names(None) == ['Subject']
        assert bare.add_entity_type('Subject') is Subject_

        # The real class binds in place of the generated one and brings its
        #  declared child along, as add_entity_type does
        assert bare.add_child_entity_type(Subject) is Subject
        assert bare.hierarchy['Subject'] == {'Session': {}}

        with pytest.raises(ValueError, match='built-in'):
            bare.add_child_entity_type('LinkEntity')

    def test_grows_by_class(self, bare):
        bare.add_entity_type(Subject)   # brings its declared child Session along

        assert bare.hierarchy['Subject'] == {'Session': {}}
        assert bare.get_entity_type('Subject') is Subject
        assert bare.get_entity_type('Session') is Session

    def test_several_roots(self, bare):
        bare.add_entity_type('Subject')
        bare.add_entity_type('Stimulus')

        assert bare.child_type_names(None) == ['Subject', 'Stimulus']
        assert [t.__name__ for t in bare.root.child_entity_types()] == ['Subject', 'Stimulus']
        bare.root.add_child(bare.get_entity_type('Stimulus'), 'grating')
        assert len(bare.get('Stimulus')) == 1

    def test_the_growth_survives_reopening(self, tmp_path):
        ent = plain(tmp_path)
        Trial = ent.add_entity_type('Trial')
        ent.root.add_child(Trial, 'trial_1', outcome='hit')
        ent.backend.close()

        reopened = entarchy.Entarchy(tmp_path.joinpath('plain').as_posix())
        assert reopened.hierarchy['Trial'] == {}
        assert reopened.get('Trial')[0]['outcome'] == 'hit'
        assert reopened.get('Trial')[0].parent.uuid == reopened.root.uuid
        reopened.backend.close()

    def test_adding_the_same_type_again_is_nothing(self, bare):
        first = bare.add_entity_type('Subject')
        assert bare.add_entity_type('Subject') is first
        assert bare.child_type_names(None) == ['Subject']

    def test_a_type_is_never_moved(self, bare):
        bare.add_entity_type('Subject')
        bare.add_entity_type('Session', parent='Subject')

        with pytest.raises(ValueError, match='already in the hierarchy under Subject, not the root'):
            bare.add_entity_type('Session')

    def test_an_unknown_parent(self, bare):
        with pytest.raises(ValueError, match='Parent type Ghost is not in the hierarchy'):
            bare.add_entity_type('Session', parent='Ghost')

    def test_built_ins_are_off_limits(self, bare):
        with pytest.raises(ValueError, match='built-in'):
            bare.add_entity_type('LinkEntity')

        class Carrier(entarchy.LinkEntity):
            pass

        with pytest.raises(TypeError, match='cannot be an entity type'):
            bare.add_entity_type(Carrier)

        with pytest.raises(ValueError, match='Parent type LinkEntity is not in the hierarchy'):
            bare.add_entity_type('Thing', parent='LinkEntity')

    def test_not_a_type(self, bare):
        with pytest.raises(TypeError, match='Entity subclass or a name'):
            bare.add_entity_type(3)


class TestBinding:
    """Classes attach to stored types by name; nothing in the store changes."""

    def test_a_stored_type_without_a_class_gets_a_generated_one(self, tmp_path):
        ent = plain(tmp_path)
        ent.add_entity_type('Subject')
        ent.backend.close()

        reopened = entarchy.Entarchy(tmp_path.joinpath('plain').as_posix())
        generated = reopened.get_entity_type('Subject')
        assert generated.__name__ == 'Subject'
        assert generated is not Subject
        assert getattr(generated, '_generated', False)
        reopened.backend.close()

    def test_a_class_binds_to_a_stored_type_of_its_name(self, tmp_path):
        ent = plain(tmp_path)
        ent.add_entity_type('Subject')
        ent.add_entity_type('Session', parent='Subject')
        ent.root.add_child(ent.get_entity_type('Subject'), 'subj_1')
        ent.backend.close()

        # LabArchy declares Subject > Session: the same shape, so it binds
        bound = LabArchy(tmp_path.joinpath('plain').as_posix())
        assert bound.get_entity_type('Subject') is Subject
        assert isinstance(bound.get(Subject)[0], Subject)
        bound.backend.close()

    def test_binding_at_open_through_entity_types(self, tmp_path):
        class Trial(entarchy.Entity):
            @property
            def shout(self):
                return self.id.upper()

        ent = plain(tmp_path)
        ent.add_entity_type('Trial')
        ent.root.add_child(ent.get_entity_type('Trial'), 'hit')
        ent.backend.close()

        reopened = entarchy.Entarchy(tmp_path.joinpath('plain').as_posix(), entity_types=[Trial])
        assert reopened.get(Trial)[0].shout == 'HIT'
        reopened.backend.close()

    def test_a_class_bound_later_replaces_the_generated_one(self, bare):
        bare.add_entity_type('Subject')
        assert getattr(bare.get_entity_type('Subject'), '_generated', False)

        assert bare.add_entity_type(Subject) is Subject
        assert bare.get_entity_type('Subject') is Subject

    def test_a_second_class_for_a_bound_name_is_refused(self, ent):
        class Subject(entarchy.Entity):   # not conftest's
            pass

        with pytest.raises(ValueError, match='already bound'):
            ent.add_entity_type(Subject)

    def test_a_declared_schema_is_added_to_a_store_that_grew_without_it(self, tmp_path):
        """What a schema package does when its hierarchy grows a level."""
        ent = plain(tmp_path)
        ent.add_entity_type('Subject')
        ent.backend.close()

        with pytest.raises(RuntimeError, match='Session'):
            LabArchy(tmp_path.joinpath('plain').as_posix())

        ent = entarchy.Entarchy(tmp_path.joinpath('plain').as_posix())
        ent.add_entity_type(Subject)   # Session follows
        ent.backend.close()

        bound = LabArchy(tmp_path.joinpath('plain').as_posix())
        assert bound.hierarchy['Subject'] == {'Session': {}}
        bound.backend.close()

    def test_a_store_with_more_than_the_class_declares_opens(self, tmp_path):
        """Compatibility is the declared tree being there, under the declared
        parents; whatever else grew on the store comes along, generated."""
        ent = plain(tmp_path)
        ent.add_entity_type(Subject)                    # Subject > Session
        ent.add_entity_type('Trial', parent='Session')
        ent.add_entity_type('Stimulus')
        ent.backend.close()

        bound = LabArchy(tmp_path.joinpath('plain').as_posix())
        assert bound.hierarchy['Subject'] == {'Session': {'Trial': {}}}
        assert bound.child_type_names(None) == ['Subject', 'Stimulus']
        assert bound.get_entity_type('Session') is Session
        assert getattr(bound.get_entity_type('Trial'), '_generated', False)
        bound.backend.close()

    def test_create_stores_the_declared_tree(self, ent):
        assert ent.hierarchy['Subject'] == {'Session': {}}
        assert ent.entity_types == (Subject, Session)

    def test_create_takes_extra_classes(self, tmp_path):
        class Stimulus(entarchy.Entity):
            pass

        base = (tmp_path / 'lab').as_posix()
        ent = LabArchy.create(base, SQLiteBackend(base, dbname='lab.db'), entity_types=[Stimulus])

        assert ent.child_type_names(None) == ['Subject', 'Stimulus']
        assert ent.get_entity_type('Stimulus') is Stimulus
        ent.backend.close()

        # Reopened without Stimulus bound, the type is still there, generated
        reopened = LabArchy(base)
        assert getattr(reopened.get_entity_type('Stimulus'), '_generated', False)
        reopened.backend.close()


class TestQueriesOnGeneratedTypes:

    def test_parent_filters_and_sorting(self, bare):
        Subject_ = bare.add_entity_type('Subject')
        Session_ = bare.add_entity_type('Session', parent=Subject_)
        with bare:
            for s in range(2):
                subject = bare.root.add_child(Subject_, f'subj_{s}', strain=['wt', 'mut'][s])
                for i in range(3):
                    subject.add_child(Session_, f'sess_{i}', index=i)

        assert len(bare.get('Session', '[Subject]strain == "mut"')) == 3
        sessions = bare.get('Session').sort_by_hierarchy()
        assert sessions.sort_keys == ['[Subject]id', 'id']
        assert bare.get('Session').dataframe_of(['index', '[Subject]strain']).shape == (6, 2)

    def test_describe_lists_the_grown_types(self, bare):
        Subject_ = bare.add_entity_type('Subject')
        bare.root.add_child(Subject_, 'subj_1')

        assert 'Subject' in bare.describe().sections['entities']['type'].tolist()


@pytest.mark.slow
class TestWorkers:
    """A worker reads the hierarchy from its own connection, so generated
    classes never have to be pickled."""

    def test_pickling_drops_the_type_map(self, bare):
        bare.add_entity_type('Subject')
        state = pickle.loads(pickle.dumps(bare)).__dict__
        assert state['_entity_map'] is None and state['_type_parents'] is None

    def test_map_async_over_a_generated_type(self, bare):
        Session_ = bare.add_entity_type('Session')
        with bare:
            for i in range(6):
                bare.root.add_child(Session_, f'sess_{i}', score=float(i))

        bare.get('Session').map_async(_mp_worker.double_score, _worker_num=2, _calibrate=False)

        out = bare.get('Session')[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()


class TestAddChildByName:

    def test_an_unbound_class_of_the_stored_name_is_accepted(self, tmp_path):
        """The store knows types by name, so the real class works on a store
        that was grown by name and opened without binding it."""
        ent = plain(tmp_path)
        ent.add_entity_type('Subject')
        ent.add_entity_type('Session', parent='Subject')

        subject = ent.root.add_child(Subject, 'subj_1')
        session = subject.add_child(Session, 'sess_1')

        assert isinstance(subject, Subject) and isinstance(session, Session)
        assert ent.get('Session')[0].parent.id == 'subj_1'
        ent.backend.close()


class TestUnboundEntities:
    """An entity is built by id alone and bound by the entarchy that takes it."""

    def test_add_child_on_the_entarchy_binds_a_root_type_under_the_root(self, bare):
        Subject_ = bare.add_entity_type('Subject')
        subject = Subject_('subject01')
        assert not subject.is_bound and subject.parent is None

        bare.add_child(subject)

        assert subject.is_bound and subject.entarchy is bare
        assert subject.parent.uuid == bare.root.uuid
        assert [s.id for s in bare.get('Subject')] == ['subject01']

    def test_attributes_written_before_binding_go_in_with_it(self, tmp_path):
        ent = plain(tmp_path)
        Subject_ = ent.add_entity_type('Subject')
        subject = Subject_('subject01')
        subject['strain'] = 'wt'
        subject['age'] = 27
        assert 'strain' in subject and 'weight' not in subject
        assert sorted(subject.keys()) == ['age', 'strain']
        assert subject[['strain', 'age']] == ('wt', 27)

        ent.add_child(subject)
        ent.backend.close()

        reopened = entarchy.Entarchy(tmp_path.joinpath('plain').as_posix())
        stored = reopened.get('Subject')[0]
        assert stored[['strain', 'age', 'id']] == ('wt', 27, 'subject01')
        reopened.backend.close()

    def test_inside_a_context(self, bare):
        Subject_ = bare.add_entity_type('Subject')
        with bare:
            for i in range(3):
                subject = Subject_(f's{i}')
                subject['index'] = i
                bare.add_child(subject)
            assert len(bare.get('Subject')) == 0

        assert sorted(s['index'] for s in bare.get('Subject')) == [0, 1, 2]

    def test_add_child_binds_too(self, bare):
        Subject_ = bare.add_entity_type('Subject')
        Session_ = bare.add_entity_type('Session', parent=Subject_)
        subject = bare.root.add_child(Subject_, 'subj_1')

        session = Session_('sess_1')
        session['rate'] = 10.0
        subject.add_child(session)

        assert session.entarchy is bare and session.parent is subject
        assert bare.get('Session')[0]['rate'] == 10.0

    def test_a_child_type_needs_its_parent(self, bare):
        Subject_ = bare.add_entity_type('Subject')
        Session_ = bare.add_entity_type('Session', parent=Subject_)

        with pytest.raises(TypeError, match='Session cannot be a child of'):
            bare.add_child(Session_('sess_1'))

        assert len(bare.get('Session')) == 0

    def test_what_an_unbound_entity_cannot_do(self, bare):
        Subject_ = bare.add_entity_type('Subject')
        subject = Subject_('subject01')

        with pytest.raises(RuntimeError, match='not bound to an entarchy yet'):
            subject.entarchy
        with pytest.raises(RuntimeError, match='nothing to commit'):
            subject.commit()
        with pytest.raises(KeyError, match='not written'):
            subject['strain']
        with pytest.raises(RuntimeError, match='entity identity'):
            subject['id'] = 'other'

    def test_bound_elsewhere(self, bare, ent):
        subject = Subject('subject01')
        bare.add_entity_type('Subject')
        bare.add_child(subject)

        with pytest.raises(ValueError, match='different entarchy'):
            ent.add_child(subject)

    def test_the_first_argument_is_the_id(self, bare):
        Subject_ = bare.add_entity_type('Subject')

        with pytest.raises(TypeError, match='the first argument is the id'):
            Subject_(bare, _id='subject01')

        with pytest.raises(TypeError):
            Subject_('subject01', bare)   # the entarchy is keyword-only

    def test_a_deferred_query_handed_to_add_child(self, bare):
        Experiment = bare.add_entity_type('Experiment')

        with pytest.raises(TypeError, match='deferred query.*get_from'):
            bare.add_child(Experiment.where('id == "Exp_01"'))

        with pytest.raises(TypeError, match='Entity subclass or an Entity instance, not str'):
            bare.add_child('Exp_01')


class TestWhereEverywhere:

    def test_the_three_entry_points_agree(self, bare):
        Subject_ = bare.add_entity_type('Subject')
        with bare:
            for i in range(4):
                bare.root.add_child(Subject_, f's{i}', index=i, strain=['wt', 'mut'][i % 2])

        by_class = Subject_.where('index > 0', strain='mut').get_from(bare)
        by_entarchy = bare.where(Subject_, 'index > 0', strain='mut')
        by_collection = bare.get(Subject_).where('index > 0', strain='mut')

        ids = lambda c: sorted(s.id for s in c)
        assert ids(by_class) == ids(by_entarchy) == ids(by_collection) == ['s1', 's3']
