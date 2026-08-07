"""The link tables and the registry of link kinds.

Kinds are data rather than Python classes, so the database is the only registry
there is and the constraints on a kind have to be recorded and checked there.
These tests cover the registry and the schema; creating links through it comes
with the write API.
"""
import uuid as uuid_module

import pytest
import sqlalchemy
from sqlalchemy.orm import Session

import entarchy
from entarchy.backend import SQLiteBackend
from entarchy.backend.sql import EntityTable, EntityTypeTable, Link, LinkTypeTable
from entarchy.core import links

from conftest import Animal, DeepArchy, Layer, Recording, Roi, make_link_row


@pytest.fixture()
def linked(deep):
    """The four-level entarchy, with a couple of kinds already registered."""
    deep.define_link_type('mean_response', Recording, Roi,
                          description='trial-averaged response')
    return deep


_make_link_row = make_link_row


class TestEndpoint:

    def test_wildcard_accepts_anything(self):
        wildcard = links.Endpoint()

        assert wildcard.is_wildcard
        assert wildcard.accepts(links.Endpoint(entity_type='Roi'))
        assert wildcard.accepts(links.Endpoint(link_type='mean_response'))

    def test_entity_endpoint_matches_by_type(self):
        endpoint = links.Endpoint(entity_type='Roi')

        assert endpoint.accepts(links.Endpoint(entity_type='Roi'))
        assert not endpoint.accepts(links.Endpoint(entity_type='Animal'))

    def test_entity_endpoint_rejects_a_link(self):
        """Every link is a LinkEntity, so an entity constraint must not match one."""
        endpoint = links.Endpoint(entity_type='Roi')

        assert not endpoint.accepts(links.Endpoint(link_type='mean_response'))

    def test_link_endpoint_matches_by_kind(self):
        endpoint = links.Endpoint(link_type='mean_response')

        assert endpoint.accepts(links.Endpoint(link_type='mean_response'))
        assert not endpoint.accepts(links.Endpoint(link_type='correlated'))

    def test_endpoint_cannot_be_both(self):
        with pytest.raises(links.LinkTypeError):
            links.Endpoint(entity_type='Roi', link_type='mean_response')


class TestOrientation:

    def test_matching_pair_is_kept(self):
        spec = links.LinkTypeSpec('mean_response',
                                  links.Endpoint(entity_type='Recording'),
                                  links.Endpoint(entity_type='Roi'))

        assert links.orientation(spec, links.Endpoint(entity_type='Recording'),
                                 links.Endpoint(entity_type='Roi')) == 'as_given'

    def test_reversed_pair_is_swapped(self):
        """With different endpoint types there is only one way it can be meant."""
        spec = links.LinkTypeSpec('mean_response',
                                  links.Endpoint(entity_type='Recording'),
                                  links.Endpoint(entity_type='Roi'))

        assert links.orientation(spec, links.Endpoint(entity_type='Roi'),
                                 links.Endpoint(entity_type='Recording')) == 'swapped'

    def test_wrong_types_raise(self):
        spec = links.LinkTypeSpec('mean_response',
                                  links.Endpoint(entity_type='Recording'),
                                  links.Endpoint(entity_type='Roi'))

        with pytest.raises(links.LinkTypeError, match='connects'):
            links.orientation(spec, links.Endpoint(entity_type='Animal'),
                              links.Endpoint(entity_type='Animal'))

    def test_same_type_endpoints_are_never_swapped(self):
        """Nothing to disambiguate, so a directed kind keeps what it was given."""
        spec = links.LinkTypeSpec('granger',
                                  links.Endpoint(entity_type='Roi'),
                                  links.Endpoint(entity_type='Roi'))

        assert links.orientation(spec, links.Endpoint(entity_type='Roi'),
                                 links.Endpoint(entity_type='Roi')) == 'as_given'


class TestCanonicalPair:

    def test_symmetric_kinds_are_ordered(self):
        spec = links.LinkTypeSpec('correlated',
                                  links.Endpoint(entity_type='Roi'),
                                  links.Endpoint(entity_type='Roi'),
                                  symmetric=True)

        assert links.canonical_pair(spec, 'bbb', 'aaa') == ('aaa', 'bbb')
        assert links.canonical_pair(spec, 'aaa', 'bbb') == ('aaa', 'bbb')

    def test_directed_kinds_keep_their_order(self):
        spec = links.LinkTypeSpec('granger',
                                  links.Endpoint(entity_type='Roi'),
                                  links.Endpoint(entity_type='Roi'))

        assert links.canonical_pair(spec, 'bbb', 'aaa') == ('bbb', 'aaa')


class TestSpecValidation:

    def test_unknown_cardinality_is_rejected(self):
        with pytest.raises(links.LinkTypeError, match='cardinality'):
            links.LinkTypeSpec('x', links.Endpoint(entity_type='Roi'),
                               links.Endpoint(entity_type='Roi'), cardinality='lots')

    def test_symmetric_requires_matching_endpoints(self):
        with pytest.raises(links.LinkTypeError, match='same'):
            links.LinkTypeSpec('x', links.Endpoint(entity_type='Recording'),
                               links.Endpoint(entity_type='Roi'), symmetric=True)


class TestRegistry:

    def test_define_and_read_back(self, deep):
        deep.define_link_type('mean_response', Recording, Roi, description='hello')

        spec = deep.get_link_type('mean_response')
        assert spec.linker.entity_type == 'Recording'
        assert spec.linked.entity_type == 'Roi'
        assert spec.symmetric is False
        assert spec.cardinality == 'sparse'
        assert spec.description == 'hello'

    def test_accepts_type_names_as_well_as_classes(self, deep):
        deep.define_link_type('by_name', 'Recording', 'Roi')

        assert deep.get_link_type('by_name').linker.entity_type == 'Recording'

    def test_unknown_type_is_rejected(self, deep):
        with pytest.raises(links.LinkTypeError, match='neither an entity type'):
            deep.define_link_type('bad', 'Neuron', Roi)

    def test_unknown_kind_returns_none(self, deep):
        assert deep.get_link_type('never_defined') is None

    def test_link_types_lists_everything(self, deep):
        deep.define_link_type('a_link', Recording, Roi)
        deep.define_link_type('b_link', Animal, Recording)

        assert [spec.name for spec in deep.link_types()] == ['a_link', 'b_link']

    def test_defining_twice_is_refused(self, deep):
        deep.define_link_type('mean_response', Recording, Roi)

        with pytest.raises(links.LinkTypeError, match='already defined'):
            deep.define_link_type('mean_response', Recording, Roi)

    def test_wildcard_endpoints(self, deep):
        deep.define_link_type('annotates', None, Roi)

        spec = deep.get_link_type('annotates')
        assert spec.linker.is_wildcard
        assert spec.linked.entity_type == 'Roi'

    def test_registry_survives_reopening(self, deep, tmp_path):
        deep.define_link_type('mean_response', Recording, Roi, description='persisted')
        path = deep.path
        deep.backend.close()

        reopened = DeepArchy(path)
        try:
            assert reopened.get_link_type('mean_response').description == 'persisted'
        finally:
            reopened.backend.close()


class TestDirectionDeclaration:

    def test_same_endpoints_require_a_declaration(self, deep):
        """Nothing else can say which end is which."""
        with pytest.raises(links.LinkTypeError, match='symmetric'):
            deep.define_link_type('correlated', Roi, Roi)

    def test_symmetric_may_be_declared(self, deep):
        deep.define_link_type('correlated', Roi, Roi, symmetric=True)

        assert deep.get_link_type('correlated').symmetric is True

    def test_directed_may_be_declared(self, deep):
        deep.define_link_type('granger', Roi, Roi, symmetric=False)

        assert deep.get_link_type('granger').symmetric is False

    def test_differing_endpoints_need_no_declaration(self, deep):
        deep.define_link_type('mean_response', Recording, Roi)

        assert deep.get_link_type('mean_response').symmetric is False


class TestLinkEndpoints:
    """A link endpoint is constrained by kind, since all links share one type."""

    def test_link_kind_as_endpoint(self, linked):
        linked.define_link_type('adaptation', 'mean_response', 'mean_response',
                                symmetric=False)

        spec = linked.get_link_type('adaptation')
        assert spec.linker.link_type == 'mean_response'
        assert spec.linker.entity_type is None

    def test_undefined_link_kind_as_endpoint_is_rejected(self, deep):
        with pytest.raises(links.LinkTypeError, match='neither an entity type'):
            deep.define_link_type('adaptation', 'not_defined_yet', 'not_defined_yet')

    def test_a_link_endpoint_does_not_match_an_entity_endpoint(self, linked):
        """The point of the second column: LinkEntity would match everything."""
        linked.define_link_type('adaptation', 'mean_response', 'mean_response',
                                symmetric=False)
        spec = linked.get_link_type('adaptation')

        assert not spec.linker.accepts(links.Endpoint(entity_type='LinkEntity'))
        assert spec.linker.accepts(links.Endpoint(link_type='mean_response'))


class TestRedefine:

    def test_redefine_with_no_links(self, deep):
        deep.define_link_type('mean_response', Recording, Roi)
        deep.redefine_link_type('mean_response', Animal, Roi, description='changed')

        spec = deep.get_link_type('mean_response')
        assert spec.linker.entity_type == 'Animal'
        assert spec.description == 'changed'

    def test_redefine_is_refused_while_links_exist(self, linked):
        recording = linked.get(Recording)[0]
        roi = linked.get(Roi)[0]
        _make_link_row(linked, 'mean_response', recording.uuid, roi.uuid)

        with pytest.raises(links.LinkTypeError, match='exist'):
            linked.redefine_link_type('mean_response', Animal, Roi)

    def test_redefine_can_drop_existing_links(self, linked):
        recording = linked.get(Recording)[0]
        roi = linked.get(Roi)[0]
        _make_link_row(linked, 'mean_response', recording.uuid, roi.uuid)

        linked.redefine_link_type('mean_response', Animal, Roi, delete_existing=True)

        assert linked.backend.count_links_of_type('mean_response') == 0
        assert linked.get_link_type('mean_response').linker.entity_type == 'Animal'

    def test_dropping_links_removes_their_carrier_entities(self, linked):
        recording = linked.get(Recording)[0]
        roi = linked.get(Roi)[0]
        link_uuid = _make_link_row(linked, 'mean_response', recording.uuid, roi.uuid)

        linked.backend.remove_links_of_type('mean_response')

        with Session(linked.backend.sql_engine) as session:
            assert session.get(EntityTable, link_uuid) is None


class TestSchemaConstraints:

    def test_a_pair_may_carry_several_kinds(self, linked):
        """The kind is part of the key, which the old schema did not allow."""
        linked.define_link_type('peak_latency', Recording, Roi)
        recording = linked.get(Recording)[0]
        roi = linked.get(Roi)[0]

        _make_link_row(linked, 'mean_response', recording.uuid, roi.uuid)
        _make_link_row(linked, 'peak_latency', recording.uuid, roi.uuid)

        assert linked.backend.count_links_of_type('mean_response') == 1
        assert linked.backend.count_links_of_type('peak_latency') == 1

    def test_the_same_kind_twice_on_one_pair_is_refused(self, linked):
        recording = linked.get(Recording)[0]
        roi = linked.get(Roi)[0]
        _make_link_row(linked, 'mean_response', recording.uuid, roi.uuid)

        with pytest.raises(sqlalchemy.exc.IntegrityError):
            _make_link_row(linked, 'mean_response', recording.uuid, roi.uuid)

    def test_link_type_must_be_registered(self, linked):
        recording = linked.get(Recording)[0]
        roi = linked.get(Roi)[0]

        with Session(linked.backend.sql_engine) as session:
            session.execute(sqlalchemy.text('PRAGMA foreign_keys=ON'))
            session.add(Link(link_uuid=str(uuid_module.uuid4()),
                             link_type='never_registered',
                             linker_uuid=recording.uuid, linked_uuid=roi.uuid))
            with pytest.raises(sqlalchemy.exc.IntegrityError):
                session.commit()

    def test_carrier_entity_is_a_child_of_the_linker(self, linked):
        """Which is what gives links archive and map_async locality."""
        recording = linked.get(Recording)[0]
        roi = linked.get(Roi)[0]
        link_uuid = _make_link_row(linked, 'mean_response', recording.uuid, roi.uuid)

        with Session(linked.backend.sql_engine) as session:
            assert session.get(EntityTable, link_uuid).parent_uuid == recording.uuid
