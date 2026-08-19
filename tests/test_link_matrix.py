"""Reading links back as a matrix.

The counterpart to link_from_matrix. What makes it worth having beyond a pivot
the caller could write is orientation: a symmetric kind stores its ends in uuid
order and a directed kind stores them the way its declaration says, so the pair
a caller asked for is not the pair the rows carry.
"""
import numpy as np
import pytest

from conftest import Animal, Layer, Recording, Roi


@pytest.fixture()
def two_subsets(deep):
    """Two disjoint sets of ROIs, one per recording, each in a known order."""
    for recording in deep.get(Recording):
        with deep:
            for index, roi in enumerate(sorted(recording.entarchy.get(
                    Roi, f'[Recording]id == "{recording.id}"'), key=lambda e: e.uuid)):
                roi['order'] = index

    a = deep.get(Roi, '[Recording]id == "rec_0"').sort('order')
    b = deep.get(Roi, '[Recording]id == "rec_1"').sort('order')

    return deep, a, b


def written_matrix(rows, columns, seed=0):
    return np.random.default_rng(seed).random((rows, columns))


class TestSymmetricAcrossTwoSubsets:

    @pytest.fixture()
    def filled(self, two_subsets):
        ent, a, b = two_subsets
        ent.define_link_type('corr', Roi, Roi, symmetric=True)
        values = written_matrix(len(a), len(b))
        ent.link_from_matrix(a, b, values, 'corr', where=lambda m: m > -1.0,
                             value_name='r')

        return ent, a, b, values

    def test_it_gives_back_what_was_written(self, filled):
        ent, a, b, values = filled
        assert np.allclose(ent.matrix_from_links(a, b, 'corr', 'r').to_numpy(), values)

    def test_the_shape_is_the_two_collections(self, filled):
        ent, a, b, values = filled
        assert ent.matrix_from_links(a, b, 'corr', 'r').shape == (len(a), len(b))

    def test_it_is_indexed_by_uuid_and_named_by_type(self, filled):
        ent, a, b, _ = filled
        matrix = ent.matrix_from_links(a, b, 'corr', 'r')

        assert list(matrix.index) == [roi.uuid for roi in a]
        assert list(matrix.columns) == [roi.uuid for roi in b]
        assert (matrix.index.name, matrix.columns.name) == ('Roi', 'Roi')

    def test_asking_the_other_way_round_transposes_it(self, filled):
        """Which end of a symmetric link is stored as the linker is an artifact
        of uuid ordering, so neither direction is the stored one."""
        ent, a, b, values = filled
        assert np.allclose(ent.matrix_from_links(b, a, 'corr', 'r').to_numpy(),
                           values.T)

    def test_the_shorthand_is_the_same_thing(self, filled):
        ent, a, b, values = filled
        assert np.allclose(a.matrix_from_links(b, 'corr', 'r').to_numpy(),
                           ent.matrix_from_links(a, b, 'corr', 'r').to_numpy())

    def test_rows_follow_the_collection_order(self, filled):
        ent, a, b, values = filled
        reversed_a = a.sort('-order')

        assert np.allclose(ent.matrix_from_links(reversed_a, b, 'corr', 'r').to_numpy(),
                           values[::-1])

    def test_a_smaller_collection_gives_a_smaller_matrix(self, filled):
        ent, a, b, values = filled
        some = a.where('order < 2').sort('order')

        matrix = ent.matrix_from_links(some, b, 'corr', 'r')
        assert matrix.shape == (2, len(b))
        assert np.allclose(matrix.to_numpy(), values[:2])


class TestSymmetricOverOneCollection:

    @pytest.fixture()
    def filled(self, two_subsets):
        ent, a, _ = two_subsets
        ent.define_link_type('corr', Roi, Roi, symmetric=True)
        values = written_matrix(len(a), len(a))
        values = (values + values.T) / 2
        ent.link_from_matrix(a, a, values, 'corr', where=lambda m: m > -1.0,
                             value_name='r')

        return ent, a, values

    def test_it_comes_back_symmetric_not_triangular(self, filled):
        """Each unordered pair is one link, so a pivot on the stored ends alone
        would fill half the matrix and leave the rest NaN."""
        ent, a, _ = filled
        read = ent.matrix_from_links(a, a, 'corr', 'r').to_numpy()

        assert np.allclose(read, read.T, equal_nan=True)

    def test_the_off_diagonal_is_what_was_written(self, filled):
        ent, a, values = filled
        read = ent.matrix_from_links(a, a, 'corr', 'r').to_numpy()
        off = ~np.eye(len(a), dtype=bool)

        assert np.allclose(read[off], values[off])

    def test_the_diagonal_is_nan(self, filled):
        """link_from_matrix does not write an entity to itself, so nothing is
        there to read - and NaN says so where a zero would be a claim."""
        ent, a, _ = filled
        read = ent.matrix_from_links(a, a, 'corr', 'r').to_numpy()

        assert np.isnan(np.diag(read)).all()


class TestDirectedKinds:

    def test_the_declared_direction_is_the_one_that_has_it(self, two_subsets):
        ent, a, b = two_subsets
        ent.define_link_type('leads_to', Roi, Roi, symmetric=False)
        values = written_matrix(len(a), len(b))
        ent.link_from_matrix(a, b, values, 'leads_to', where=lambda m: m > -1.0,
                             value_name='w')

        assert np.allclose(ent.matrix_from_links(a, b, 'leads_to', 'w').to_numpy(),
                           values)
        assert np.isnan(ent.matrix_from_links(b, a, 'leads_to', 'w').to_numpy()).all()

    def test_endpoints_of_different_types_may_be_given_either_way(self, two_subsets):
        """links_to() stores the pair the way the kind declares it, so asking
        for the rows the other way round has to transpose rather than scramble."""
        ent, a, _ = two_subsets
        layers = ent.get(Layer, '[Recording]id == "rec_0"').sort('id')
        ent.define_link_type('response', Layer, Roi)

        values = written_matrix(len(layers), len(a), seed=1)
        ent.link_from_matrix(layers, a, values, 'response', where=lambda m: m > -1.0,
                             value_name='amp')

        assert np.allclose(
            ent.matrix_from_links(layers, a, 'response', 'amp').to_numpy(), values)
        assert np.allclose(
            ent.matrix_from_links(a, layers, 'response', 'amp').to_numpy(), values.T)

    def test_the_axes_are_named_for_their_own_types(self, two_subsets):
        ent, a, _ = two_subsets
        layers = ent.get(Layer, '[Recording]id == "rec_0"').sort('id')
        ent.define_link_type('response', Layer, Roi)
        ent.link_from_matrix(layers, a, written_matrix(len(layers), len(a)),
                             'response', where=lambda m: m > -1.0, value_name='amp')

        matrix = ent.matrix_from_links(layers, a, 'response', 'amp')
        assert (matrix.index.name, matrix.columns.name) == ('Layer', 'Roi')


class TestGapsAndFilters:

    @pytest.fixture()
    def thresholded(self, two_subsets):
        ent, a, b = two_subsets
        ent.define_link_type('corr', Roi, Roi, symmetric=True)
        values = written_matrix(len(a), len(b))
        ent.link_from_matrix(a, b, values, 'corr', where=lambda m: m > 0.5,
                             value_name='r')

        return ent, a, b, values

    def test_a_pair_with_no_link_is_nan(self, thresholded):
        ent, a, b, values = thresholded
        read = ent.matrix_from_links(a, b, 'corr', 'r').to_numpy()

        assert np.isnan(read[values <= 0.5]).all()
        assert np.allclose(read[values > 0.5], values[values > 0.5])

    def test_a_filter_expression_narrows_it_further(self, thresholded):
        ent, a, b, values = thresholded
        read = ent.matrix_from_links(a, b, 'corr', 'r', 'r > 0.8').to_numpy()

        assert np.isfinite(read).sum() == (values > 0.8).sum()

    def test_a_filter_leaves_the_shape_alone(self, thresholded):
        """The rows are the collections, not whatever survived the filter."""
        ent, a, b, _ = thresholded
        assert ent.matrix_from_links(a, b, 'corr', 'r', 'r > 0.8').shape == (len(a), len(b))

    def test_a_kind_with_no_links_is_the_right_shape_of_nothing(self, two_subsets):
        ent, a, b = two_subsets
        ent.define_link_type('never', Roi, Roi, symmetric=True)

        empty = ent.matrix_from_links(a, b, 'never', 'r')
        assert empty.shape == (len(a), len(b))
        assert np.isnan(empty.to_numpy()).all()

    def test_an_attribute_the_links_do_not_carry_says_so(self, thresholded):
        ent, a, b, _ = thresholded
        with pytest.raises(AttributeError, match='not found'):
            ent.matrix_from_links(a, b, 'corr', 'no_such_value')


class TestValuesThatAreNotNumbers:

    def test_text_survives_rather_than_being_refused(self, two_subsets):
        """A link may carry a label as well as a number, and the float cast is
        an accommodation rather than a requirement."""
        import pandas as pd

        ent, a, b = two_subsets
        ent.define_link_type('named', Roi, Roi, symmetric=True)
        ent.link_from_frame(pd.DataFrame({
            'linker_uuid': [a[0].uuid],
            'linked_uuid': [b[0].uuid],
            'label': ['strong'],
        }), 'named')

        matrix = ent.matrix_from_links(a, b, 'named', 'label')
        assert matrix.iloc[0, 0] == 'strong'
        assert pd.isna(matrix.iloc[1, 1])


class TestTheEndpointQuery:

    def test_it_refuses_a_collection_that_is_not_of_links(self, two_subsets):
        ent, a, _ = two_subsets
        with pytest.raises(TypeError, match='links_to'):
            ent.backend.get_link_endpoints(a)

    def test_it_carries_the_collection_filter(self, two_subsets):
        ent, a, b = two_subsets
        ent.define_link_type('corr', Roi, Roi, symmetric=True)
        values = written_matrix(len(a), len(b))
        ent.link_from_matrix(a, b, values, 'corr', where=lambda m: m > -1.0,
                             value_name='r')

        pairs = a.links_to(b, 'corr', 'r > 0.5')
        endpoints = ent.backend.get_link_endpoints(pairs)

        assert len(endpoints) == len(pairs) == int((values > 0.5).sum())
        assert all(len(row) == 3 for row in endpoints)
