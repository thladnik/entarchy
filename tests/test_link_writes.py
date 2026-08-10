"""Creating links.

The guards get as much attention as the happy path, because the failure this
design is most exposed to is quiet: nothing stops a pairwise write from being
every pair, and a runaway one succeeds slowly rather than raising.
"""
import numpy as np
import pandas as pd
import pytest

from entarchy.core import links

from conftest import Animal, DeepArchy, Layer, Recording, Roi


@pytest.fixture()
def wired(deep):
    """The four-level entarchy with handles on a recording and its ROIs."""
    recording = sorted(deep.get(Recording), key=lambda e: e.id)[0]
    rois = sorted(recording.entarchy.get(Roi), key=lambda e: e.uuid)
    return deep, recording, rois


def _spec(name='k', linker='Roi', linked='Roi', **kwargs):
    return links.LinkTypeSpec(name, links.Endpoint(entity_type=linker),
                              links.Endpoint(entity_type=linked), **kwargs)


class TestSingleLink:

    def test_creates_and_carries_attributes(self, wired):
        ent, recording, rois = wired

        with ent:
            link = ent.link(recording, rois[0], 'mean_response', mean_dff=0.42)

        assert link.link_type == 'mean_response'
        assert link.linker.uuid == recording.uuid
        assert link.linked.uuid == rois[0].uuid
        assert link['mean_dff'] == 0.42

    def test_kind_is_registered_on_first_use(self, wired):
        ent, recording, rois = wired
        assert ent.get_link_type('mean_response') is None

        with ent:
            ent.link(recording, rois[0], 'mean_response')

        spec = ent.get_link_type('mean_response')
        assert spec.linker.entity_type == 'Recording'
        assert spec.linked.entity_type == 'Roi'

    def test_wrong_endpoints_are_refused_after_registration(self, wired):
        ent, recording, rois = wired
        with ent:
            ent.link(recording, rois[0], 'mean_response')

        animal = ent.get(Animal)[0]
        with pytest.raises(links.LinkTypeError, match='connects'):
            with ent:
                ent.link(animal, animal, 'mean_response')

    def test_same_type_endpoints_need_defining_first(self, wired):
        """Nothing can infer the direction, so first use cannot register it."""
        ent, recording, rois = wired

        with pytest.raises(links.LinkTypeError, match='direction cannot be inferred'):
            with ent:
                ent.link(rois[0], rois[1], 'correlated')

    def test_arguments_may_be_the_wrong_way_round(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('mean_response', Recording, Roi)

        with ent:
            link = ent.link(rois[0], recording, 'mean_response')

        assert link.linker.uuid == recording.uuid
        assert link.linked.uuid == rois[0].uuid

    def test_linking_twice_returns_the_same_link(self, wired):
        ent, recording, rois = wired

        with ent:
            first = ent.link(recording, rois[0], 'mean_response', mean_dff=1.0)
        with ent:
            second = ent.link(recording, rois[0], 'mean_response')

        assert first.uuid == second.uuid
        assert ent.backend.count_links_of_type('mean_response') == 1

    def test_linking_twice_within_one_block(self, wired):
        ent, recording, rois = wired

        with ent:
            first = ent.link(recording, rois[0], 'mean_response')
            second = ent.link(recording, rois[0], 'mean_response')

        assert first.uuid == second.uuid
        assert ent.backend.count_links_of_type('mean_response') == 1

    def test_bulk_sees_links_queued_in_the_same_block(self, wired):
        ent, recording, rois = wired

        with ent:
            ent.link(recording, rois[0], 'mean_response')
            result = ent.link_from_frame(pd.DataFrame({
                'linker_uuid': [recording.uuid, recording.uuid],
                'linked_uuid': [rois[0].uuid, rois[1].uuid],
            }), 'mean_response')

        assert result.already_present == 1
        assert result.created == 1
        assert ent.backend.count_links_of_type('mean_response') == 2

    def test_a_pair_may_carry_two_kinds(self, wired):
        ent, recording, rois = wired

        with ent:
            ent.link(recording, rois[0], 'mean_response', value=1.0)
            ent.link(recording, rois[0], 'peak_latency', value=2.0)

        assert ent.get_link(recording, rois[0], 'mean_response')['value'] == 1.0
        assert ent.get_link(recording, rois[0], 'peak_latency')['value'] == 2.0

    def test_carrier_is_a_child_of_the_linker(self, wired):
        ent, recording, rois = wired

        with ent:
            link = ent.link(recording, rois[0], 'mean_response')

        assert link.parent.uuid == recording.uuid

    def test_works_outside_a_context(self, wired):
        ent, recording, rois = wired

        link = ent.link(recording, rois[0], 'mean_response', mean_dff=0.5)

        assert ent.get_link(recording, rois[0], 'mean_response').uuid == link.uuid


class TestSymmetry:

    def test_symmetric_links_are_found_from_either_end(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        with ent:
            ent.link(rois[1], rois[0], 'correlated', r=0.8)

        assert ent.get_link(rois[0], rois[1], 'correlated')['r'] == 0.8
        assert ent.get_link(rois[1], rois[0], 'correlated')['r'] == 0.8

    def test_symmetric_links_are_stored_once(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        with ent:
            ent.link(rois[0], rois[1], 'correlated')
        with ent:
            ent.link(rois[1], rois[0], 'correlated')

        assert ent.backend.count_links_of_type('correlated') == 1

    def test_symmetric_links_are_stored_once_within_one_block(self, wired):
        """The second call cannot query for the first: it is not committed yet."""
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        with ent:
            first = ent.link(rois[0], rois[1], 'correlated', r=0.9)
            second = ent.link(rois[1], rois[0], 'correlated')

        assert first.uuid == second.uuid
        assert ent.backend.count_links_of_type('correlated') == 1
        assert ent.get_link(rois[0], rois[1], 'correlated')['r'] == 0.9

    def test_directed_kinds_keep_both_directions(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('granger', Roi, Roi, symmetric=False)

        with ent:
            ent.link(rois[0], rois[1], 'granger', f=4.2)
            ent.link(rois[1], rois[0], 'granger', f=0.3)

        assert ent.backend.count_links_of_type('granger') == 2
        assert ent.get_link(rois[0], rois[1], 'granger')['f'] == 4.2
        assert ent.get_link(rois[1], rois[0], 'granger')['f'] == 0.3

    def test_other_end_traverses_a_symmetric_link(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        with ent:
            link = ent.link(rois[0], rois[1], 'correlated')

        assert link.other_end(rois[0]).uuid == rois[1].uuid
        assert link.other_end(rois[1]).uuid == rois[0].uuid

    def test_other_end_rejects_a_stranger(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        with ent:
            link = ent.link(rois[0], rois[1], 'correlated')

        with pytest.raises(ValueError, match='not an endpoint'):
            link.other_end(rois[2])


class TestLinksBetweenLinks:

    def test_a_link_may_link_two_links(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('mean_response', Recording, Roi)

        with ent:
            first = ent.link(recording, rois[0], 'mean_response')
            second = ent.link(recording, rois[1], 'mean_response')

        ent.define_link_type('adaptation', 'mean_response', 'mean_response',
                             symmetric=False)
        with ent:
            link = ent.link(second, first, 'adaptation', ratio=0.68)

        assert link['ratio'] == 0.68
        assert link.linker.uuid == second.uuid

    def test_an_entity_is_refused_where_a_link_is_declared(self, wired):
        """The reason link endpoints are constrained by kind, not entity type."""
        ent, recording, rois = wired
        ent.define_link_type('mean_response', Recording, Roi)
        ent.define_link_type('adaptation', 'mean_response', 'mean_response',
                             symmetric=False)

        with ent:
            response = ent.link(recording, rois[0], 'mean_response')

        with pytest.raises(links.LinkTypeError, match='connects'):
            with ent:
                ent.link(response, rois[1], 'adaptation')

    def test_the_wrong_link_kind_is_refused(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('mean_response', Recording, Roi)
        ent.define_link_type('peak_latency', Recording, Roi)
        ent.define_link_type('adaptation', 'mean_response', 'mean_response',
                             symmetric=False)

        with ent:
            response = ent.link(recording, rois[0], 'mean_response')
            latency = ent.link(recording, rois[1], 'peak_latency')

        with pytest.raises(links.LinkTypeError, match='connects'):
            with ent:
                ent.link(response, latency, 'adaptation')


class TestReading:

    def test_entity_links_in_both_directions(self, wired):
        ent, recording, rois = wired

        with ent:
            ent.link(recording, rois[0], 'mean_response')
            ent.link(recording, rois[1], 'mean_response')

        assert len(recording.links('mean_response')) == 2
        assert len(rois[0].links('mean_response')) == 1

    def test_direction_filters(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('granger', Roi, Roi, symmetric=False)

        with ent:
            ent.link(rois[0], rois[1], 'granger')
            ent.link(rois[2], rois[0], 'granger')

        assert len(rois[0].links('granger', direction='out')) == 1
        assert len(rois[0].links('granger', direction='in')) == 1
        assert len(rois[0].links('granger')) == 2

    def test_link_types_on_an_entity(self, wired):
        ent, recording, rois = wired

        with ent:
            ent.link(recording, rois[0], 'mean_response')
            ent.link(recording, rois[0], 'peak_latency')

        assert rois[0].link_types() == ['mean_response', 'peak_latency']

    def test_link_counts_on_an_entity(self, wired):
        ent, recording, rois = wired

        with ent:
            ent.link(recording, rois[0], 'mean_response')
            ent.link(recording, rois[1], 'mean_response')
            ent.link(recording, rois[0], 'peak_latency')

        assert recording.link_counts() == {'mean_response': 2, 'peak_latency': 1}
        assert rois[1].link_counts() == {'mean_response': 1}

    def test_link_counts_see_both_ends(self, wired):
        """Counted from either end, so an entity that is only ever the linked
        end does not look unlinked."""
        ent, recording, rois = wired
        ent.define_link_type('granger', Roi, Roi, symmetric=False)

        with ent:
            ent.link(rois[0], rois[1], 'granger')
            ent.link(rois[2], rois[0], 'granger')

        assert rois[0].link_counts() == {'granger': 2}
        assert rois[1].link_counts() == {'granger': 1}

    def test_link_counts_is_empty_without_links(self, wired):
        ent, recording, rois = wired
        assert rois[0].link_counts() == {}

    def test_get_link_returns_none_for_an_unknown_kind(self, wired):
        ent, recording, rois = wired

        assert ent.get_link(recording, rois[0], 'never_defined') is None

    def test_there_is_no_matmul_operator(self, wired):
        """Links are created by name, not by an operator; @ is left unimplemented."""
        ent, recording, rois = wired

        with pytest.raises(TypeError, match='unsupported operand'):
            recording @ rois[0]


class TestBulkFrame:

    def test_creates_links_and_attributes(self, wired):
        ent, recording, rois = wired

        result = ent.link_from_frame(pd.DataFrame({
            'linker_uuid': [recording.uuid] * 3,
            'linked_uuid': [roi.uuid for roi in rois[:3]],
            'mean_dff': [1.0, 2.0, 3.0],
        }), 'mean_response')

        assert result.created == 3
        assert ent.backend.count_links_of_type('mean_response') == 3
        assert ent.get_link(recording, rois[0], 'mean_response')['mean_dff'] == 1.0
        assert ent.get_link(recording, rois[2], 'mean_response')['mean_dff'] == 3.0

    def test_rerunning_creates_nothing(self, wired):
        ent, recording, rois = wired
        frame = pd.DataFrame({'linker_uuid': [recording.uuid] * 3,
                              'linked_uuid': [roi.uuid for roi in rois[:3]],
                              'mean_dff': [1.0, 2.0, 3.0]})

        ent.link_from_frame(frame, 'mean_response')
        result = ent.link_from_frame(frame, 'mean_response')

        assert result.created == 0
        assert result.already_present == 3
        assert ent.backend.count_links_of_type('mean_response') == 3

    def test_duplicates_within_the_input_are_dropped(self, wired):
        ent, recording, rois = wired

        result = ent.link_from_frame(pd.DataFrame({
            'linker_uuid': [recording.uuid] * 3,
            'linked_uuid': [rois[0].uuid, rois[0].uuid, rois[1].uuid],
            'mean_dff': [1.0, 9.0, 2.0],
        }), 'mean_response')

        assert result.duplicates_dropped == 1
        assert result.created == 2
        # The first occurrence wins
        assert ent.get_link(recording, rois[0], 'mean_response')['mean_dff'] == 1.0

    def test_missing_column_is_reported(self, wired):
        ent, recording, rois = wired

        with pytest.raises(ValueError, match='linked_uuid'):
            ent.link_from_frame(pd.DataFrame({'linker_uuid': [recording.uuid]}),
                                'mean_response')

    def test_unknown_uuid_is_reported(self, wired):
        ent, recording, rois = wired

        with pytest.raises(ValueError, match='not entities'):
            ent.link_from_frame(pd.DataFrame({
                'linker_uuid': [recording.uuid],
                'linked_uuid': ['00000000-0000-0000-0000-000000000000'],
            }), 'mean_response')

    def test_empty_frame_is_a_no_op(self, wired):
        ent, recording, rois = wired

        result = ent.link_from_frame(
            pd.DataFrame({'linker_uuid': [], 'linked_uuid': []}), 'mean_response')

        assert result.created == 0

    def test_dry_run_writes_nothing(self, wired):
        ent, recording, rois = wired

        result = ent.link_from_frame(pd.DataFrame({
            'linker_uuid': [recording.uuid] * 3,
            'linked_uuid': [roi.uuid for roi in rois[:3]],
        }), 'mean_response', dry_run=True)

        assert result.dry_run
        assert result.created == 3
        assert ent.backend.count_links_of_type('mean_response') == 0

    def test_symmetric_pairs_are_deduplicated(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        result = ent.link_from_frame(pd.DataFrame({
            'linker_uuid': [rois[0].uuid, rois[1].uuid],
            'linked_uuid': [rois[1].uuid, rois[0].uuid],
            'r': [0.9, 0.9],
        }), 'correlated')

        assert result.duplicates_dropped == 1
        assert result.created == 1


class TestGuards:
    """check_write_size is exercised directly; building 100k links to test a
    guard against building 100k links would be perverse."""

    def test_confirm_count_must_match(self):
        with pytest.raises(links.LinkDensityError, match='does not match'):
            links.check_write_size(_spec(), count=10, linker_count=10, linked_count=10,
                                   confirm_count=11)

    def test_correct_confirm_count_passes(self):
        links.check_write_size(_spec(), count=10, linker_count=10, linked_count=10,
                               confirm_count=10)

    def test_volume_needs_confirmation(self):
        """Sparse but enormous: only the count ceiling should object."""
        count = links.MAX_LINKS_WITHOUT_CONFIRMATION + 1

        with pytest.raises(links.LinkDensityError, match='confirm_count'):
            links.check_write_size(_spec(), count=count, linker_count=count,
                                   linked_count=count)

    def test_a_dense_write_is_refused(self):
        with pytest.raises(links.LinkDensityError, match='every possible pair'):
            links.check_write_size(_spec(), count=90_000, linker_count=300,
                                   linked_count=300)

    def test_density_is_ignored_for_small_writes(self):
        """A recording linked to each of its five ROIs is 100% dense and fine."""
        links.check_write_size(_spec(), count=5, linker_count=1, linked_count=5)

    def test_dense_cardinality_clears_the_density_guard(self):
        links.check_write_size(_spec(cardinality='dense'), count=90_000,
                               linker_count=300, linked_count=300,
                               confirm_count=90_000)

    def test_density_guard_uses_a_registered_spec(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        # Small in reality, but the guard is driven by the numbers it is given
        with pytest.raises(links.LinkDensityError, match='every possible pair'):
            links.check_write_size(ent.get_link_type('correlated'), count=35_000,
                                   linker_count=200, linked_count=200)

    def test_density_exactly_at_the_threshold_passes(self):
        """Half the pairs is the boundary, and the boundary is allowed."""
        links.check_write_size(_spec(), count=20_000, linker_count=200, linked_count=200)


class TestCardinality:

    def test_one_per_linker_is_enforced(self, wired):
        ent, recording, rois = wired
        other = [r for r in ent.get(Recording) if r.uuid != recording.uuid][0]
        ent.define_link_type('preferred', Roi, Recording, cardinality='one_per_linker')

        with ent:
            ent.link(rois[0], recording, 'preferred')

        with pytest.raises(links.LinkCardinalityError, match='one_per_linker'):
            with ent:
                ent.link(rois[0], other, 'preferred')

    def test_relinking_the_same_pair_is_still_idempotent(self, wired):
        """The same pair is the same link, so it is not a second one."""
        ent, recording, rois = wired
        ent.define_link_type('preferred', Roi, Recording, cardinality='one_per_linker')

        with ent:
            first = ent.link(rois[0], recording, 'preferred')
        with ent:
            second = ent.link(rois[0], recording, 'preferred')

        assert first.uuid == second.uuid

    def test_one_per_linker_allows_one_each(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('preferred', Roi, Recording, cardinality='one_per_linker')

        with ent:
            for roi in rois[:3]:
                ent.link(roi, recording, 'preferred')

        assert ent.backend.count_links_of_type('preferred') == 3

    def test_bulk_respects_one_per_linker(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('preferred', Recording, Roi, cardinality='one_per_linker')

        with pytest.raises(links.LinkCardinalityError, match='more than one'):
            ent.link_from_frame(pd.DataFrame({
                'linker_uuid': [recording.uuid, recording.uuid],
                'linked_uuid': [rois[0].uuid, rois[1].uuid],
            }), 'preferred')


class TestFromMatrix:

    def test_threshold_selects_pairs(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        matrix = np.zeros((len(rois), len(rois)))
        matrix[0, 1] = matrix[1, 0] = 0.9
        matrix[2, 3] = matrix[3, 2] = 0.2

        result = ent.link_from_matrix(rois, rois, matrix, 'correlated',
                                      where=lambda v: np.abs(v) > 0.5, value_name='r')

        assert result.created == 1
        assert ent.get_link(rois[0], rois[1], 'correlated')['r'] == 0.9

    def test_symmetric_takes_each_pair_once(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        matrix = np.ones((len(rois), len(rois)))
        result = ent.link_from_matrix(rois, rois, matrix, 'correlated',
                                      where=lambda v: v > 0.5, value_name='r',
                                      confirm_count=len(rois) * (len(rois) - 1) // 2)

        assert result.created == len(rois) * (len(rois) - 1) // 2

    def test_predicate_is_required(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        with pytest.raises(ValueError, match='where'):
            ent.link_from_matrix(rois, rois, np.ones((len(rois), len(rois))),
                                 'correlated', where=None)

    def test_shape_is_checked(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        with pytest.raises(ValueError, match='matrix is'):
            ent.link_from_matrix(rois, rois, np.ones((2, 2)), 'correlated',
                                 where=lambda v: v > 0)

    def test_scalar_predicate_also_works(self, wired):
        """A per-element callable is vectorised rather than rejected."""
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        matrix = np.zeros((len(rois), len(rois)))
        matrix[0, 1] = matrix[1, 0] = 0.9

        result = ent.link_from_matrix(rois, rois, matrix, 'correlated',
                                      where=lambda v: bool(abs(v) > 0.5), value_name='r')

        assert result.created == 1

    def test_diagonal_is_excluded(self, wired):
        ent, recording, rois = wired
        ent.define_link_type('correlated', Roi, Roi, symmetric=True)

        matrix = np.eye(len(rois))
        result = ent.link_from_matrix(rois, rois, matrix, 'correlated',
                                      where=lambda v: v > 0.5)

        assert result.created == 0


class TestTransactional:

    def test_a_failure_leaves_no_links(self, wired):
        ent, recording, rois = wired

        with pytest.raises(RuntimeError, match='deliberate'):
            with ent:
                ent.link(recording, rois[0], 'mean_response')
                raise RuntimeError('deliberate')

        assert ent.backend.count_links_of_type('mean_response') == 0

    def test_links_and_carriers_land_together(self, wired):
        ent, recording, rois = wired

        with ent:
            ent.link(recording, rois[0], 'mean_response')
            ent.link(recording, rois[1], 'mean_response')

        rows = ent.backend.get_links_of_type('mean_response')
        assert len(rows) == 2
        for row in rows:
            assert ent.get_entity_by_uuid(row['link_uuid']) is not None
