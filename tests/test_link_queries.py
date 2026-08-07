"""Querying links.

Filtering by the attributes of a link's endpoints is what makes links worth
having: without it a link is a row you can only reach by already knowing one of
its ends. A bare name is an attribute of the link itself, and `@` addresses an
endpoint.
"""
import numpy as np
import pandas as pd
import pytest

from entarchy.core import links, query
from entarchy.core.entity import LinkCollection

from conftest import Animal, DeepArchy, Layer, Recording, Roi


@pytest.fixture()
def responses(deep):
    """One recording linked to every ROI, with values to filter on."""
    animal = deep.get(Animal)[0]
    recording = sorted(deep.get(Recording), key=lambda e: e.id)[0]
    rois = sorted(deep.get(Roi), key=lambda e: e.id)

    with deep:
        animal['strain'] = 'wildtype'
        recording['imaging_rate'] = 10.0
        for index, roi in enumerate(rois):
            roi['index'] = index
            roi['good'] = (index % 2 == 0)
            roi['quality'] = ['poor', 'fair', 'good'][index % 3]

    deep.define_link_type('mean_response', Recording, Roi, cardinality='dense')
    deep.link_from_frame(pd.DataFrame({
        'linker_uuid': [recording.uuid] * len(rois),
        'linked_uuid': [roi.uuid for roi in rois],
        'mean_dff': [0.1 * (index + 1) for index in range(len(rois))],
    }), 'mean_response')

    return deep, recording, rois


@pytest.fixture()
def correlations(deep):
    """A symmetric kind over ROIs of one layer."""
    layer = sorted(deep.get(Layer), key=lambda e: e.id)[0]
    rois = sorted(layer.entarchy.get(Roi, f'[Layer]id == "{layer.id}"'),
                  key=lambda e: e.uuid)

    with deep:
        for index, roi in enumerate(rois):
            roi['good'] = (index == 0)

    deep.define_link_type('correlated', Roi, Roi, symmetric=True)
    deep.link_from_frame(pd.DataFrame({
        'linker_uuid': [rois[0].uuid, rois[0].uuid],
        'linked_uuid': [rois[1].uuid, rois[2].uuid],
        'r': [0.9, 0.7],
    }), 'correlated')

    return deep, rois


class TestTokenizer:

    def test_endpoint_identifier(self):
        tokens = query.tokenize('@Roi.has_receptive_field == True')

        assert tokens[0] == ('IDENT', '@Roi.has_receptive_field')

    def test_endpoint_with_ancestor(self):
        tokens = query.tokenize('@linker.[Recording]imaging_rate > 8.0')

        assert tokens[0] == ('IDENT', '@linker.[Recording]imaging_rate')

    def test_endpoint_with_relative_parent(self):
        tokens = query.tokenize('@linked.../depth == 15.0')

        assert tokens[0] == ('IDENT', '@linked.../depth')

    def test_grouped_attribute_after_endpoint(self):
        tokens = query.tokenize('@Roi.s2p/npix > 40')

        assert tokens[0] == ('IDENT', '@Roi.s2p/npix')

    def test_ordinary_identifiers_are_unaffected(self):
        assert query.tokenize('../depth == 1')[0] == ('IDENT', '../depth')
        assert query.tokenize('[Animal]strain == "wt"')[0] == ('IDENT', '[Animal]strain')


class TestCollectionBasics:

    def test_links_returns_a_link_collection(self, responses):
        ent, recording, rois = responses
        collection = ent.links('mean_response')

        assert isinstance(collection, LinkCollection)
        assert collection.link_type == 'mean_response'
        assert len(collection) == len(rois)

    def test_unknown_kind_is_reported(self, responses):
        ent, recording, rois = responses

        with pytest.raises(links.LinkTypeError, match='not defined'):
            ent.links('never_defined')

    def test_iteration_yields_link_entities(self, responses):
        ent, recording, rois = responses

        for link in ent.links('mean_response'):
            assert link.link_type == 'mean_response'
            assert link.linker.uuid == recording.uuid

    def test_dataframe_of_link_attributes(self, responses):
        ent, recording, rois = responses

        frame = ent.links('mean_response').dataframe_of(['mean_dff'])

        assert frame.shape == (len(rois), 1)
        assert frame['mean_dff'].max() == pytest.approx(0.1 * len(rois))

    def test_repr_names_the_kind(self, responses):
        ent, recording, rois = responses

        assert "LinkCollection('mean_response'" in repr(ent.links('mean_response'))


class TestOwnAttributes:

    def test_comparison(self, responses):
        ent, recording, rois = responses

        selected = ent.links('mean_response', 'mean_dff > 0.35')

        assert len(selected) == len([r for r in rois]) - 3

    def test_keyword_equality(self, responses):
        ent, recording, rois = responses

        assert len(ent.links('mean_response', mean_dff=0.1)) == 1

    def test_exist(self, responses):
        ent, recording, rois = responses

        assert len(ent.links('mean_response', 'EXIST(mean_dff)')) == len(rois)
        assert len(ent.links('mean_response', 'EXIST(never_written)')) == 0


class TestEndpointByType:

    def test_filters_on_the_linked_end(self, responses):
        ent, recording, rois = responses

        selected = ent.links('mean_response', '@Roi.good == True')

        assert len(selected) == len([r for i, r in enumerate(rois) if i % 2 == 0])

    def test_filters_on_the_linker_end(self, responses):
        ent, recording, rois = responses

        assert len(ent.links('mean_response', '@Recording.imaging_rate > 8.0')) == len(rois)
        assert len(ent.links('mean_response', '@Recording.imaging_rate > 20.0')) == 0

    def test_combined_with_an_own_attribute(self, responses):
        ent, recording, rois = responses

        selected = ent.links('mean_response',
                             '@Roi.good == True AND mean_dff > 0.25')

        expected = [index for index, _ in enumerate(rois)
                    if index % 2 == 0 and 0.1 * (index + 1) > 0.25]
        assert len(selected) == len(expected)

    def test_in_operator(self, responses):
        ent, recording, rois = responses

        assert len(ent.links('mean_response', '@Roi.index IN (0, 2)')) == 2

    def test_string_comparison(self, responses):
        ent, recording, rois = responses

        selected = ent.links('mean_response', '@Roi.quality == "good"')

        assert len(selected) == len([i for i in range(len(rois)) if i % 3 == 2])

    def test_unknown_type_is_reported(self, responses):
        ent, recording, rois = responses

        with pytest.raises(ValueError, match='connects'):
            len(ent.links('mean_response', '@Animal.strain == "wildtype"'))


class TestEndpointByRole:

    def test_linker_and_linked(self, responses):
        ent, recording, rois = responses

        assert len(ent.links('mean_response', '@linker.imaging_rate > 8.0')) == len(rois)
        assert len(ent.links('mean_response', '@linked.index IN (0, 1)')) == 2

    def test_either_matches_one_end(self, responses):
        ent, recording, rois = responses

        # imaging_rate only exists on the recording, index only on the ROIs
        assert len(ent.links('mean_response', '@either.imaging_rate > 8.0')) == len(rois)
        assert len(ent.links('mean_response', '@either.index == 0')) == 1

    def test_both_requires_each_end(self, responses):
        ent, recording, rois = responses

        # Nothing has imaging_rate on both ends
        assert len(ent.links('mean_response', '@both.imaging_rate > 8.0')) == 0

    def test_unknown_role_is_reported(self, responses):
        ent, recording, rois = responses

        with pytest.raises(ValueError, match='connects'):
            len(ent.links('mean_response', '@sideways.index == 0'))

    def test_a_forgotten_dot_is_reported_clearly(self, responses):
        """'@linker' without an attribute is a plausible typo, not a parse error."""
        ent, recording, rois = responses

        with pytest.raises(ValueError, match='Malformed endpoint reference'):
            len(ent.links('mean_response', '@linker == 1'))


class TestAncestorTraversal:

    def test_named_ancestor_of_an_endpoint(self, responses):
        ent, recording, rois = responses

        assert len(ent.links('mean_response',
                             '@linker.[Animal]strain == "wildtype"')) == len(rois)
        assert len(ent.links('mean_response',
                             '@linker.[Animal]strain == "mutant"')) == 0

    def test_ancestor_of_the_linked_end(self, responses):
        ent, recording, rois = responses

        assert len(ent.links('mean_response',
                             '@Roi.[Animal]strain == "wildtype"')) == len(rois)

    def test_relative_parent_of_an_endpoint(self, responses):
        ent, recording, rois = responses

        # A ROI's parent is its layer
        selected = ent.links('mean_response', '@linked.../depth == 0.0')
        assert len(selected) > 0

    def test_non_ancestor_is_reported(self, responses):
        ent, recording, rois = responses

        with pytest.raises(ValueError, match='not an ancestor'):
            len(ent.links('mean_response', '@Roi.[Phase]index == 1'))


class TestSymmetricKinds:

    def test_type_addressing_means_either_end(self, correlations):
        ent, rois = correlations

        # Only rois[0] is good, and it is an endpoint of both links
        assert len(ent.links('correlated', '@Roi.good == True')) == 2

    def test_role_addressing_is_refused(self, correlations):
        ent, rois = correlations

        with pytest.raises(ValueError, match='symmetric'):
            len(ent.links('correlated', '@linker.good == True'))

        with pytest.raises(ValueError, match='symmetric'):
            len(ent.links('correlated', '@linked.good == True'))

    def test_both_still_works(self, correlations):
        ent, rois = correlations

        assert len(ent.links('correlated', '@both.good == True')) == 0

    def test_own_attributes_are_unaffected(self, correlations):
        ent, rois = correlations

        assert len(ent.links('correlated', 'r > 0.8')) == 1


class TestBooleanCombinations:

    def test_not_with_an_endpoint(self, responses):
        ent, recording, rois = responses

        negated = ent.links('mean_response', 'NOT(@Roi.good == True)')

        assert len(negated) == len([i for i in range(len(rois)) if i % 2 == 1])

    def test_or_across_endpoint_and_own(self, responses):
        ent, recording, rois = responses

        selected = ent.links('mean_response',
                             '@Roi.index == 0 OR mean_dff > 0.55')

        assert len(selected) >= 1

    def test_exist_on_an_endpoint(self, responses):
        ent, recording, rois = responses

        assert len(ent.links('mean_response', 'EXIST(@Roi.index)')) == len(rois)
        assert len(ent.links('mean_response', 'EXIST(@Roi.nothing_here)')) == 0

    def test_parenthesised_grouping(self, responses):
        ent, recording, rois = responses

        selected = ent.links('mean_response',
                             '(@Roi.good == True OR mean_dff > 0.55) AND EXIST(mean_dff)')

        assert len(selected) > 0


class TestSetOperations:
    """Derived collections must keep the link kind."""

    def test_where_keeps_the_kind(self, responses):
        ent, recording, rois = responses

        narrowed = ent.links('mean_response').where('@Roi.good == True')

        assert isinstance(narrowed, LinkCollection)
        assert narrowed.link_type == 'mean_response'

    def test_intersection_keeps_the_kind(self, responses):
        ent, recording, rois = responses

        narrowed = ent.links('mean_response') & '@Roi.good == True'

        assert isinstance(narrowed, LinkCollection)
        assert len(narrowed) == len([i for i in range(len(rois)) if i % 2 == 0])

    def test_union_keeps_the_kind(self, responses):
        ent, recording, rois = responses

        combined = (ent.links('mean_response', '@Roi.index == 0')
                    | '@Roi.index == 1')

        assert isinstance(combined, LinkCollection)
        assert len(combined) == 2

    def test_inversion_keeps_the_kind(self, responses):
        ent, recording, rois = responses

        inverted = ~ent.links('mean_response', '@Roi.good == True')

        assert isinstance(inverted, LinkCollection)
        assert len(inverted) == len([i for i in range(len(rois)) if i % 2 == 1])


class TestEndpointsOutsideLinkCollections:

    def test_endpoint_syntax_is_refused_on_an_entity_collection(self, responses):
        ent, recording, rois = responses

        with pytest.raises(ValueError, match='rather than of links'):
            len(ent.get(Roi, '@Recording.imaging_rate > 8.0'))


class TestWritingThroughACollection:

    def test_update_writes_link_attributes(self, responses):
        ent, recording, rois = responses
        collection = ent.links('mean_response')
        uuids = [uuid for uuid, _ in
                 ent.backend.get_collection_parent_uuids(collection)]

        collection.update(pd.DataFrame({'checked': [True] * len(uuids)}, index=uuids))

        assert len(ent.links('mean_response', 'checked == True')) == len(rois)


@pytest.mark.slow
class TestParallel:

    def test_map_async_over_links(self, responses):
        import _mp_worker

        ent, recording, rois = responses

        ent.links('mean_response').map_async(_mp_worker.scale_link_value,
                                             factor=3.0, _worker_num=2,
                                             _calibrate=False)

        frame = ent.links('mean_response').dataframe_of(['mean_dff', 'scaled',
                                                         'linked_index'])
        assert len(frame) == len(rois)
        np.testing.assert_allclose(frame['scaled'], frame['mean_dff'] * 3.0)
