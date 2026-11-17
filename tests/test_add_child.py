"""Entity.add_child: the one place the hierarchy is checked on write."""
import pytest

import entarchy
from conftest import Animal, Layer, LabArchy, Recording, Roi, Session, Subject


class TestCreatesTheChild:

    def test_from_a_class_and_an_id(self, ent):
        subject = ent.root.add_child(Subject, 'subj_1')
        session = subject.add_child(Session, 'sess_1')

        assert isinstance(session, Session)
        assert session.id == 'sess_1'
        assert session.parent is subject
        assert len(ent.get(Session)) == 1

    def test_attributes_go_in_with_it(self, ent):
        subject = ent.root.add_child(Subject, 'subj_1', strain='wt', age=12)

        fetched = ent.get(Subject)[0]
        assert fetched['strain'] == 'wt'
        assert fetched['age'] == 12

    def test_the_root_takes_the_root_types(self, ent):
        subject = ent.root.add_child(Subject, 'subj_1')

        assert subject.parent.uuid == ent.root.uuid
        assert ent.root.child_entity_types() == (Subject,)

    def test_from_an_instance(self, ent):
        subject = ent.root.add_child(Subject, 'subj_1')
        session = Session(ent, _id='sess_1')

        returned = subject.add_child(session)

        assert returned is session
        assert session.parent is subject
        assert [s.id for s in ent.get(Session)] == ['sess_1']

    def test_an_instance_built_with_this_parent_is_fine(self, ent):
        subject = ent.root.add_child(Subject, 'subj_1')
        session = Session(ent, _id='sess_1', _parent=subject)

        subject.add_child(session)

        assert len(ent.get(Session)) == 1

    def test_inside_a_context_nothing_is_written_until_exit(self, ent):
        with ent:
            subject = ent.root.add_child(Subject, 'subj_1', strain='wt')
            assert len(ent.get(Subject)) == 0

        assert len(ent.get(Subject)) == 1
        assert ent.get(Subject)[0]['strain'] == 'wt'

    def test_survives_reopening(self, tmp_path):
        base = (tmp_path / 'archy').as_posix()
        ent = LabArchy.create(base, entarchy.backend.SQLiteBackend(base, dbname='t.db'))
        ent.root.add_child(Subject, 'subj_1').add_child(Session, 'sess_1', rate=10.0)
        ent.backend.close()

        reopened = LabArchy(base)
        session = reopened.get(Session)[0]
        assert session.parent.id == 'subj_1'
        assert session['rate'] == 10.0
        reopened.backend.close()


class TestTheTypeCheck:

    def test_an_unregistered_child_type(self, ent):
        subject = ent.root.add_child(Subject, 'subj_1')

        with pytest.raises(TypeError, match='Subject cannot be a child of Subject, '
                                            'whose child entity types are Session'):
            subject.add_child(Subject, 'subj_2')

    def test_a_leaf_has_no_child_types(self, ent):
        session = ent.root.add_child(Subject, 'subj_1').add_child(Session, 'sess_1')

        with pytest.raises(TypeError, match='Session has no child entity types'):
            session.add_child(Session, 'sess_2')

    def test_only_a_root_type_goes_under_the_root(self, ent):
        with pytest.raises(TypeError, match='Session cannot be a child of EntarchyEntity'):
            ent.root.add_child(Session, 'sess_1')

    def test_a_type_from_another_hierarchy(self, ent):
        subject = ent.root.add_child(Subject, 'subj_1')

        with pytest.raises(TypeError, match='Recording cannot be a child of Subject'):
            subject.add_child(Recording, 'rec_1')

    def test_the_instance_form_is_checked_too(self, ent):
        subject = ent.root.add_child(Subject, 'subj_1')
        another = Subject(ent, _id='subj_2')

        with pytest.raises(TypeError, match='cannot be a child'):
            subject.add_child(another)

    def test_a_rejected_child_is_not_written(self, ent):
        subject = ent.root.add_child(Subject, 'subj_1')

        with pytest.raises(TypeError):
            subject.add_child(Subject, 'subj_2')

        assert len(ent.get(Subject)) == 1

    def test_something_that_is_not_an_entity(self, ent):
        with pytest.raises(TypeError, match='not an Entity subclass'):
            ent.root.add_child(dict, 'x')

        with pytest.raises(TypeError, match='Entity subclass or an Entity instance'):
            ent.root.add_child('Subject', 'x')

    def test_links_and_analyses_are_not_children(self, ent):
        subject = ent.root.add_child(Subject, 'subj_1')

        with pytest.raises(TypeError):
            subject.add_child(entarchy.LinkEntity, 'l')
        with pytest.raises(TypeError):
            subject.add_child(entarchy.AnalysisEntity, 'a')

    def test_a_subclass_of_a_registered_type_is_not_registered(self, ent):
        """The backend stores types by class name and knows only the names in
        the hierarchy; a subclass would fail deep in the write with a KeyError."""
        class LongSession(Session):
            pass

        subject = ent.root.add_child(Subject, 'subj_1')

        with pytest.raises(TypeError, match='LongSession cannot be a child of Subject'):
            subject.add_child(LongSession, 'sess_1')


class TestArguments:

    def test_a_class_needs_an_id(self, ent):
        with pytest.raises(ValueError, match='An id is needed'):
            ent.root.add_child(Subject)

    def test_an_instance_takes_no_id(self, ent):
        with pytest.raises(ValueError, match='has its id already'):
            ent.root.add_child(Subject(ent, _id='subj_1'), 'subj_2')

    def test_an_instance_with_another_parent(self, ent):
        a = ent.root.add_child(Subject, 'a')
        b = ent.root.add_child(Subject, 'b')
        session = Session(ent, _id='sess_1', _parent=a)

        with pytest.raises(ValueError, match='is a child of .*a.*, not of'):
            b.add_child(session)

    def test_an_instance_already_added(self, ent):
        subject = ent.root.add_child(Subject, 'subj_1')
        with ent:
            session = subject.add_child(Session, 'sess_1')

            with pytest.raises(ValueError, match='already been added'):
                subject.add_child(session)

    def test_an_instance_from_another_entarchy(self, ent, deep):
        subject = ent.root.add_child(Subject, 'subj_1')
        foreign = Session(deep, _id='sess_1')

        with pytest.raises(ValueError, match='different entarchy'):
            subject.add_child(foreign)


class TestDeepHierarchy:

    def test_builds_the_fixture_tree(self, deep):
        animal = deep.get(Animal)[0]
        recording = animal.add_child(Recording, 'rec_2', rate=12.0)
        layer = recording.add_child(Layer, 'plane0', depth=0.0)
        roi = layer.add_child(Roi, 'roi_0', index=0)

        assert len(deep.get(Recording)) == 3
        assert deep.get(Roi, '[Recording]id == "rec_2"')[0].uuid == roi.uuid
        assert roi.parent.parent.parent.uuid == animal.uuid
