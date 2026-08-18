"""describe(): what an entity or collection holds, without reading it."""
import numpy as np
import pandas as pd
import pytest

from conftest import Animal, DeepArchy, Layer, Recording, Roi
from entarchy import MediaFile
from entarchy.backend import SQLiteBackend
from entarchy.core import describe as describe_mod


@pytest.fixture()
def described(tmp_path):
    """One of each thing a description has a section for."""
    base = (tmp_path / 'described').as_posix()
    ent = DeepArchy.create(base, SQLiteBackend(base, dbname='described.db'))

    video = tmp_path / 'behaviour.avi'
    video.write_bytes(b'FAKE-AVI' + bytes(range(256)) * 200)

    with ent:
        animal = Animal(ent, _id='animal_1', _parent=ent.root)
        ent.add_new_entity(animal)
        animal['strain'] = 'wildtype'

        recording = Recording(ent, _id='rec_0', _parent=animal)
        ent.add_new_entity(recording)
        recording['rate'] = 10.0
        recording['times'] = np.linspace(0, 1, 5000)

        layer = Layer(ent, _id='plane0', _parent=recording)
        ent.add_new_entity(layer)
        layer['depth'] = 0.0

        for index in range(6):
            roi = Roi(ent, _id=f'Roi_{index}', _parent=layer)
            ent.add_new_entity(roi)
            roi['index'] = index
            roi['good'] = (index % 2 == 0)
            roi['dff'] = np.random.rand(500)
            if index < 3:
                roi['sparse'] = float(index)

    recording.set_media('video', str(video))
    ent.commit()

    ent.define_link_type('mean_response', linker='Roi', linked='Roi', symmetric=False)
    rois = sorted(ent.get(Roi), key=lambda r: r['index'])
    with ent:
        for other in rois[1:]:
            ent.link(rois[0], other, 'mean_response', mean_dff=0.5, p_value=0.01)

    yield ent
    ent.backend.close()


def row_for(frame, name, column='name'):
    match = frame[frame[column] == name]
    assert len(match) == 1, f'{name} not in {list(frame[column])}'
    return match.iloc[0]


class TestHelpers:

    @pytest.mark.parametrize('count,expected', [
        (0, '0 B'), (512, '512 B'), (1024, '1.0 kB'),
        (1536, '1.5 kB'), (1024 ** 2, '1.0 MB'), (5 * 1024 ** 3, '5.0 GB'),
    ])
    def test_human_bytes(self, count, expected):
        assert describe_mod.human_bytes(count) == expected

    def test_short_value_cuts_long_text(self):
        out = describe_mod.short_value('x' * 200)
        assert len(out) == 60
        assert out.endswith('...')

    def test_short_value_stays_encodable_on_a_legacy_console(self):
        """A description is printed to whatever console is there, and a Windows
        one on cp1252 raises on an ellipsis character."""
        describe_mod.short_value('x' * 200).encode('cp1252')

    def test_short_value_leaves_short_text(self):
        assert describe_mod.short_value(12) == '12'


class TestRangeEndpoints:
    """_range_of on its own, for the cases a backend cannot easily be made to
    produce on demand."""

    def blank(self, **kwargs):
        entry = {'min': None, 'max': None, 'distinct': 0,
                 'nan': 0, 'plus_inf': 0, 'minus_inf': 0}
        entry.update(kwargs)
        return entry

    def test_one_types_ends_are_the_databases_and_not_re_derived(self):
        """MySQL 8 calls 'abc' the least of ['abc', 'ABC', 'Zeta'] and 'Zeta'
        the greatest. Handed just those two, Python's byte order puts 'Zeta'
        first - so re-minimising here would report the range backwards."""
        distribution = {('label', 'str'): self.blank(min='abc', max='Zeta',
                                                     distinct=3)}
        row = describe_mod._range_of('label', ['str'], distribution)

        assert row['min'] == 'abc'
        assert row['max'] == 'Zeta'

    def test_several_numeric_types_are_combined(self):
        distribution = {('n', 'int'): self.blank(min=3, max=9, distinct=2),
                        ('n', 'float'): self.blank(min=0.5, max=2.5, distinct=2)}
        row = describe_mod._range_of('n', ['float', 'int'], distribution)

        assert row['min'] == 0.5
        assert row['max'] == 9
        assert row['distinct'] == 4

    def test_a_long_text_endpoint_is_cut(self):
        """value_str is a TEXT column, so one value could take the table with
        it. Numbers are left alone, being their own length."""
        distribution = {('n', 'str'): self.blank(min='x' * 500, max='y',
                                                 distinct=2)}
        row = describe_mod._range_of('n', ['str'], distribution)

        assert len(row['min']) == 60 and row['min'].endswith('...')


class TestDescriptionObject:

    def test_empty_sections_are_left_out(self):
        description = describe_mod.Description(
            'thing', {}, {'attributes': pd.DataFrame({'name': ['a']}),
                          'links': pd.DataFrame()})
        assert 'attributes' in description
        assert 'links' not in description
        assert list(description.sections) == ['attributes']

    def test_missing_section_reads_as_empty(self):
        description = describe_mod.Description('thing', {}, {})
        assert description.links.empty
        assert description.media.empty

    def test_indexing_a_missing_section_says_what_is_there(self):
        description = describe_mod.Description(
            'thing', {}, {'attributes': pd.DataFrame({'name': ['a']})})
        with pytest.raises(KeyError, match='attributes'):
            description['links']

    def test_repr_survives_a_broken_section(self):
        """A description is reached for when something is already confusing."""
        class Exploding(pd.DataFrame):
            def to_string(self, *args, **kwargs):
                raise RuntimeError('boom')

        description = describe_mod.Description(
            'thing', {}, {'attributes': Exploding({'name': ['a']})})
        assert 'could not be rendered' in repr(description)

    def test_notes_are_shown(self):
        description = describe_mod.Description('thing', {}, {}, notes=['capped'])
        assert 'capped' in repr(description)
        assert description.notes == ['capped']


class TestEntityDescribe:

    def test_headline_identifies_the_entity(self, described):
        roi = described.get(Roi, 'index == 0')[0]
        description = roi.describe()

        assert description.headline['type'] == 'Roi'
        assert description.headline['id'] == 'Roi_0'
        assert description.headline['uuid'] == roi.uuid
        assert description.headline['path'].endswith('plane0/Roi_0')

    def test_attributes_carry_type_and_size(self, described):
        roi = described.get(Roi, 'index == 0')[0]
        frame = roi.describe().attributes

        assert row_for(frame, 'index')['type'] == 'int'
        assert row_for(frame, 'dff')['type'] == 'blob'
        assert row_for(frame, 'dff')['bytes'].endswith('kB')

    def test_scalar_values_are_inlined_and_blobs_are_not(self, described):
        roi = described.get(Roi, 'index == 0')[0]
        frame = roi.describe().attributes

        assert row_for(frame, 'index')['value'] == '0'
        assert row_for(frame, 'good')['value'] == 'True'
        assert row_for(frame, 'dff')['value'] == ''

    def test_blobs_are_never_read(self, described):
        """The whole point: a description of an entity holding hundreds of
        megabytes must not load them."""
        roi = described.get(Roi, 'index == 0')[0]
        asked = []

        original = described.backend.get_entity_attributes

        def recording_call(entity, names):
            asked.extend(names)
            return original(entity, names)

        described.backend.get_entity_attributes = recording_call
        try:
            roi.describe()
        finally:
            described.backend.get_entity_attributes = original

        assert 'dff' not in asked

    def test_values_can_be_left_out(self, described):
        roi = described.get(Roi, 'index == 0')[0]
        frame = roi.describe(values=False).attributes

        assert set(frame['value']) == {''}
        assert row_for(frame, 'index')['type'] == 'int'

    def test_bookkeeping_is_not_listed_as_data(self, described):
        roi = described.get(Roi, 'index == 0')[0]
        names = list(roi.describe().attributes['name'])

        assert 'id' not in names
        assert 'uuid' not in names
        assert 'index' in names

    def test_links_say_what_they_carry(self, described):
        roi = described.get(Roi, 'index == 0')[0]
        frame = roi.describe().links

        row = row_for(frame, 'mean_response', column='kind')
        assert row['links'] == 5
        assert row['carries'] == 'mean_dff, p_value'

    def test_link_bookkeeping_is_not_carried(self, described):
        """id and uuid are on a link made one at a time and not on one written
        in bulk, so neither belongs in what the kind carries."""
        roi = described.get(Roi, 'index == 0')[0]
        carries = row_for(roi.describe().links, 'mean_response', column='kind')['carries']

        assert 'uuid' not in carries
        assert 'id' not in carries

    def test_an_entity_without_links_has_no_links_section(self, described):
        assert 'links' not in described.get(Recording)[0].describe()

    def test_media_is_named_and_checked_for_presence(self, described):
        recording = described.get(Recording)[0]
        frame = recording.describe().media

        row = row_for(frame, 'video')
        assert row['media type'] == 'video/avi'
        assert row['present']
        assert 'verified' not in frame.columns

    def test_media_is_not_verified_unless_asked(self, described):
        recording = described.get(Recording)[0]
        frame = recording.describe(verify=True).media

        assert row_for(frame, 'video')['verified']

    def test_media_reads_as_media_among_the_attributes(self, described):
        """Saying blob of it would send a reader looking for an array."""
        recording = described.get(Recording)[0]
        frame = recording.describe().attributes

        assert row_for(frame, 'video')['type'] == 'media'
        assert row_for(frame, 'times')['type'] == 'blob'

    def test_children_are_counted_by_type(self, described):
        frame = described.get(Recording)[0].describe().children
        assert row_for(frame, 'Layer', column='type')['count'] == 1

        frame = described.get(Layer)[0].describe().children
        assert row_for(frame, 'Roi', column='type')['count'] == 6

    def test_links_are_not_counted_as_children(self, described):
        """A link's carrier is parented to its linker, which keeps the tree
        valid and would otherwise tell a ROI it has five children."""
        roi = described.get(Roi, 'index == 0')[0]
        description = roi.describe()

        assert 'LinkEntity' not in list(description.children.get('type', []))
        assert 'children' not in description

    def test_ancestry_runs_outermost_first(self, described):
        roi = described.get(Roi, 'index == 0')[0]
        frame = roi.describe().ancestry

        assert list(frame['type']) == ['Animal', 'Recording', 'Layer']
        assert list(frame['id']) == ['animal_1', 'rec_0', 'plane0']

    def test_the_root_is_not_an_ancestor_worth_listing(self, described):
        """Every entity has it, so a row saying so carries nothing."""
        animal = described.get(Animal)[0]
        assert 'ancestry' not in animal.describe()

    def test_renders_as_text_and_as_html(self, described):
        description = described.get(Roi, 'index == 0')[0].describe()

        text = repr(description)
        assert 'Roi_0' in text and 'attributes' in text and 'mean_response' in text

        markup = description._repr_html_()
        assert '<table' in markup and 'mean_response' in markup


class TestCollectionDescribe:

    def test_headline_counts_and_names_the_order(self, described):
        description = described.get(Roi).sort('index').describe()

        assert description.headline['entity type'] == 'Roi'
        assert description.headline['entities'] == 6
        assert description.headline['order'] == 'index'

    def test_coverage_is_the_column_one_entity_cannot_show(self, described):
        frame = described.get(Roi).describe().attributes

        assert row_for(frame, 'index')['entities'] == '6 / 6'
        assert row_for(frame, 'sparse')['entities'] == '3 / 6'

    def test_sizes_are_totals_across_the_collection(self, described):
        frame = described.get(Roi).describe().attributes
        assert row_for(frame, 'index')['bytes'] == '48 B'

    def test_several_types_for_one_name_are_shown(self, described):
        with described:
            rois = sorted(described.get(Roi), key=lambda r: r['index'])
            rois[0]['mixed'] = 1
            rois[1]['mixed'] = 1.5

        frame = described.get(Roi).describe().attributes
        assert row_for(frame, 'mixed')['type'] == '{float, int}'

    def test_children_are_counted_over_the_whole_collection(self, described):
        frame = described.get(Layer).describe().children
        assert row_for(frame, 'Roi', column='type')['count'] == 6

    def test_links_are_counted_once_not_per_endpoint(self, described):
        """Every one of the five links has both ends inside this collection."""
        frame = described.get(Roi).describe().links
        assert row_for(frame, 'mean_response', column='kind')['links'] == 5

    def test_link_counting_can_be_turned_off(self, described):
        assert 'links' not in described.get(Roi).describe(links=False)

    def test_an_empty_collection_describes_without_raising(self, described):
        description = described.get(Roi, 'index == 99').describe()
        assert description.headline['entities'] == 0
        assert 'nothing to show' in repr(description)


class TestPreviewLeavesBlobsAlone:

    def test_blobs_are_left_out_by_default(self, described):
        frame = described.get(Roi).preview(3)

        assert 'dff' not in frame.columns
        assert 'index' in frame.columns

    def test_what_was_left_out_is_recorded(self, described):
        frame = described.get(Roi).preview(3)
        assert frame.attrs['blobs_omitted'] == ['dff']

    def test_what_was_left_out_is_printed(self, described, capsys):
        described.get(Roi).preview(3)
        assert 'left out 1 blob attribute' in capsys.readouterr().out

    def test_blobs_can_be_asked_for(self, described):
        frame = described.get(Roi).preview(3, blobs=True)

        assert 'dff' in frame.columns
        assert len(frame['dff'].iloc[0]) == 500

    def test_named_attributes_are_taken_as_asked(self, described):
        """Naming a blob is asking for it."""
        frame = described.get(Roi).preview(2, attribute_names=['dff'])

        assert list(frame.columns) == ['dff']
        assert frame.attrs['blobs_omitted'] == []

    def test_an_entity_with_no_blobs_prints_nothing(self, described, capsys):
        described.get(Layer).preview(1)
        assert 'left out' not in capsys.readouterr().out


@pytest.fixture()
def special_floats(tmp_path):
    """NaN and infinity, which are stored as a flag with a null value column
    and are therefore invisible to MIN, MAX and COUNT(DISTINCT)."""
    base = (tmp_path / 'floats').as_posix()
    ent = DeepArchy.create(base, SQLiteBackend(base, dbname='floats.db'))

    with ent:
        animal = Animal(ent, _id='animal_1', _parent=ent.root)
        ent.add_new_entity(animal)
        recording = Recording(ent, _id='rec_0', _parent=animal)
        ent.add_new_entity(recording)
        layer = Layer(ent, _id='plane0', _parent=recording)
        ent.add_new_entity(layer)

        #        snr: three finite values and a NaN
        #       gain: three finite values and both infinities
        snr = [1.0, 2.0, float('nan'), 2.0, 4.0]
        gain = [1.0, 2.0, float('inf'), float('-inf'), 5.0]

        for index in range(len(snr)):
            roi = Roi(ent, _id=f'Roi_{index}', _parent=layer)
            ent.add_new_entity(roi)
            roi['index'] = index
            roi['snr'] = snr[index]
            roi['gain'] = gain[index]

    yield ent
    ent.backend.close()


@pytest.fixture()
def empty(tmp_path):
    base = (tmp_path / 'empty').as_posix()
    ent = DeepArchy.create(base, SQLiteBackend(base, dbname='empty.db'))
    yield ent
    ent.backend.close()


class TestCollectionDistribution:

    def test_it_is_off_by_default(self, described):
        frame = described.get(Roi).describe().attributes
        assert 'min' not in frame.columns
        assert 'distinct' not in frame.columns

    def test_asking_adds_the_three_columns(self, described):
        frame = described.get(Roi).describe(distribution=True).attributes
        assert list(frame.columns) == ['name', 'type', 'entities', 'bytes',
                                       'min', 'max', 'distinct']

    def test_an_integer_range_is_the_values(self, described):
        row = row_for(described.get(Roi).describe(distribution=True).attributes,
                      'index')

        assert row['min'] == 0
        assert row['max'] == 5
        assert row['distinct'] == 6

    def test_a_range_endpoint_is_a_value_not_a_rendering_of_one(self, described):
        """So it can be compared against rather than parsed back."""
        row = row_for(described.get(Roi).describe(distribution=True).attributes,
                      'index')
        assert row['max'] - row['min'] == 5

    def test_booleans_have_a_range_too(self, described):
        row = row_for(described.get(Roi).describe(distribution=True).attributes,
                      'good')
        assert row['distinct'] == 2

    def test_text_has_a_range(self, described):
        with described:
            for index, roi in enumerate(sorted(described.get(Roi),
                                               key=lambda r: r['index'])):
                roi['label'] = f'cell {index}'

        row = row_for(described.get(Roi).describe(distribution=True).attributes,
                      'label')
        assert row['min'] == 'cell 0'
        assert row['max'] == 'cell 5'

    def test_a_range_covers_only_the_entities_that_have_the_name(self, described):
        """`sparse` is on three of six ROIs, and a range over the other three
        would have to invent something for them."""
        row = row_for(described.get(Roi).describe(distribution=True).attributes,
                      'sparse')

        assert row['entities'] == '3 / 6'
        assert row['distinct'] == 3

    def test_a_blob_has_no_range(self, described):
        row = row_for(described.get(Roi).describe(distribution=True).attributes,
                      'dff')

        assert row['min'] == ''
        assert row['max'] == ''
        assert row['distinct'] == ''

    def test_a_constant_attribute_shows_as_one_distinct_value(self, described):
        with described:
            for roi in described.get(Roi):
                roi['footprint'] = 1

        row = row_for(described.get(Roi).describe(distribution=True).attributes,
                      'footprint')
        assert row['min'] == row['max'] == 1
        assert row['distinct'] == 1

    def test_the_range_is_of_the_collection_not_the_entarchy(self, described):
        frame = (described.get(Roi, 'index < 3').describe(distribution=True)
                 .attributes)
        assert row_for(frame, 'index')['max'] == 2

    def test_a_name_stored_as_two_types_gets_one_range(self, described):
        with described:
            rois = sorted(described.get(Roi), key=lambda r: r['index'])
            rois[0]['mixed'] = 1
            rois[1]['mixed'] = 1.5

        row = row_for(described.get(Roi).describe(distribution=True).attributes,
                      'mixed')

        assert row['type'] == '{float, int}'
        assert row['min'] == 1
        assert row['max'] == 1.5
        assert row['distinct'] == 2

    def test_a_name_stored_as_text_and_number_has_no_range(self, described):
        """Python refuses to compare them, and inventing an order here would be
        worse than a blank."""
        with described:
            rois = sorted(described.get(Roi), key=lambda r: r['index'])
            rois[0]['muddle'] = 1
            rois[1]['muddle'] = 'one'

        row = row_for(described.get(Roi).describe(distribution=True).attributes,
                      'muddle')

        assert row['type'] == '{int, str}'
        assert row['min'] == '' and row['max'] == ''
        assert row['distinct'] == 2

    def test_an_empty_collection_still_describes(self, described):
        description = described.get(Roi, 'index == 99').describe(distribution=True)
        assert description.headline['entities'] == 0


class TestSpecialFloatsInARange:
    """NaN and infinity are stored as a flag with a null value column, so a
    range that trusted MIN and MAX would quietly be a range over the rest."""

    def test_infinity_is_the_end_of_the_range(self, special_floats):
        row = row_for(special_floats.get(Roi).describe(distribution=True).attributes,
                      'gain')

        assert row['min'] == float('-inf')
        assert row['max'] == float('inf')

    def test_both_infinities_count_as_values(self, special_floats):
        """1.0, 2.0, 5.0 and the two infinities."""
        row = row_for(special_floats.get(Roi).describe(distribution=True).attributes,
                      'gain')
        assert row['distinct'] == 5

    def test_nan_is_kept_out_of_the_range(self, special_floats):
        row = row_for(special_floats.get(Roi).describe(distribution=True).attributes,
                      'snr')

        assert row['min'] == 1.0
        assert row['max'] == 4.0

    def test_nan_is_counted_as_a_value(self, special_floats):
        """1.0, 2.0, 4.0 and the NaN; the second 2.0 is not a fourth."""
        row = row_for(special_floats.get(Roi).describe(distribution=True).attributes,
                      'snr')
        assert row['distinct'] == 4

    def test_a_note_says_where_nan_is(self, special_floats):
        description = special_floats.get(Roi).describe(distribution=True)

        assert any('NaN' in note and 'snr' in note for note in description.notes)
        assert not any('gain' in note for note in description.notes)

    def test_an_attribute_that_is_only_infinity_has_it_at_both_ends(self, described):
        """MIN and MAX both come back null, so the range has to be rebuilt from
        the flags rather than filled in around them."""
        with described:
            described.get(Roi, 'index == 0')[0]['ceiling'] = float('inf')

        row = row_for(described.get(Roi).describe(distribution=True).attributes,
                      'ceiling')

        assert row['min'] == float('inf')
        assert row['max'] == float('inf')
        assert row['distinct'] == 1

    def test_an_attribute_that_is_only_nan_is_counted_without_a_range(self, described):
        with described:
            described.get(Roi, 'index == 0')[0]['broken'] = float('nan')

        row = row_for(described.get(Roi).describe(distribution=True).attributes,
                      'broken')

        assert row['min'] == '' and row['max'] == ''
        assert row['distinct'] == 1

    def test_no_note_when_nothing_holds_nan(self, described):
        description = described.get(Roi).describe(distribution=True)
        assert not any('NaN' in note for note in description.notes)


class TestEntarchyDescribe:

    def test_headline_says_what_is_there(self, described):
        headline = described.describe().headline

        assert headline['backend'] == 'SQLiteBackend'
        assert headline['path'] == described.path
        assert headline['links'] == 5
        # the root, an animal, a recording, a layer and six ROIs
        assert headline['entities'] == 10

    def test_entities_are_counted_by_type(self, described):
        frame = described.describe().entities
        row = row_for(frame, 'Roi', column='type')

        assert row['entities'] == 6
        assert row['parent'] == 'Layer'

    def test_types_run_in_hierarchy_order(self, described):
        """Alphabetically it would put Roi above Recording and tell a stranger
        nothing about which contains which."""
        types = list(described.describe().entities['type'])

        assert (types.index('Animal') < types.index('Recording')
                < types.index('Layer') < types.index('Roi'))

    def test_link_carriers_are_not_in_the_entity_census(self, described):
        """Every link is one, so they would outnumber the entities that were
        actually written."""
        assert 'LinkEntity' not in list(described.describe().entities['type'])

    def test_a_declared_type_with_no_entities_is_left_out(self, empty):
        assert 'Roi' not in list(empty.describe().entities.get('type', []))

    def test_link_kinds_say_what_they_join(self, described):
        row = row_for(described.describe().links, 'mean_response', column='kind')

        assert row['between'] == 'Roi -> Roi'
        assert row['cardinality'] == 'sparse'
        assert row['links'] == 5

    def test_a_symmetric_kind_reads_as_undirected(self, described):
        described.define_link_type('overlaps', linker='Roi', linked='Roi',
                                   symmetric=True)
        row = row_for(described.describe().links, 'overlaps', column='kind')

        assert row['between'] == 'Roi -- Roi'

    def test_a_registered_but_unused_kind_is_still_shown(self, described):
        """Kinds are invented at runtime and the registry is the only schema
        there is, so a declared kind with no links is worth meeting."""
        described.define_link_type('never_used', linker='Layer', linked='Roi')
        row = row_for(described.describe().links, 'never_used', column='kind')

        assert row['links'] == 0
        assert row['bytes'] == '0 B'

    def test_storage_ranks_the_largest_first(self, described):
        raw = described.backend.get_attribute_storage()
        by_size = {(type_name, name): size
                   for type_name, name, _, _, size in raw}

        frame = described.describe(largest=None).storage
        ranked = [by_size[(row['entity type'], row['name'])]
                  for _, row in frame.iterrows()]

        assert ranked == sorted(ranked, reverse=True)
        assert len(ranked) > 1

    def test_a_media_file_is_ranked_at_its_size_on_disk(self, described):
        """Its bytes are in the entarchy directory rather than in a row, and
        the question the section answers is where the bytes went."""
        row = row_for(described.describe().storage, 'video')

        assert row['entity type'] == 'Recording'
        assert row['bytes'].endswith('kB')

    def test_storage_names_the_type_the_attribute_is_on(self, described):
        frame = described.describe().storage
        assert row_for(frame, 'dff')['entity type'] == 'Roi'
        assert row_for(frame, 'dff')['entities'] == 6

    def test_storage_is_capped_and_says_how_many_were_cut(self, described):
        description = described.describe(largest=2)

        assert len(description.storage) == 2
        assert any('not shown' in note for note in description.notes)

    def test_storage_can_be_asked_for_whole(self, described):
        description = described.describe(largest=None)

        assert len(description.storage) > 2
        assert not any('not shown' in note for note in description.notes)

    def test_the_two_sections_account_for_every_byte(self, described):
        """The entity census leaves link carriers out and the links section
        picks them up, so between them they have to come to the total - which
        is the reason the census leaves them out rather than hiding them."""
        storage = described.backend.get_attribute_storage()
        total = sum(size for *_, size in storage)

        entity_bytes = sum(size for type_name, *_, size in storage
                           if type_name != describe_mod.LINK_ENTITY_TYPE)
        link_bytes = sum(entry['bytes'] for entry
                         in described.backend.get_link_type_totals().values())

        assert entity_bytes + link_bytes == total
        assert total > 0

    def test_an_empty_entarchy_describes_without_raising(self, empty):
        description = empty.describe()

        assert description.headline['entities'] == 1  # the root
        assert description.headline['links'] == 0

    def test_renders_as_text_and_as_html(self, described):
        description = described.describe()

        text = repr(description)
        assert 'Roi' in text and 'mean_response' in text and 'storage' in text

        markup = description._repr_html_()
        assert '<table' in markup and 'mean_response' in markup


class TestEntarchyStorageQueries:

    def test_entity_counts_cover_every_type_written(self, described):
        counts = described.backend.count_entities_by_type()

        assert counts['Roi'] == 6
        assert counts['Layer'] == 1
        assert counts[describe_mod.LINK_ENTITY_TYPE] == 5

    def test_link_totals_count_links_not_endpoints(self, described):
        totals = described.backend.get_link_type_totals()

        assert totals['mean_response']['links'] == 5
        assert totals['mean_response']['bytes'] > 0

    def test_a_kind_whose_links_carry_nothing_still_counts(self, described):
        """An inner join would drop it, and a kind with no attributes is a
        perfectly ordinary way to record that two things are related."""
        described.define_link_type('touches', linker='Roi', linked='Roi',
                                   symmetric=False)
        rois = sorted(described.get(Roi), key=lambda r: r['index'])
        with described:
            described.link(rois[1], rois[2], 'touches')

        assert described.backend.get_link_type_totals()['touches']['links'] == 1

    def test_storage_is_grouped_by_type_and_name(self, described):
        storage = described.backend.get_attribute_storage()
        rows = [row for row in storage if row[0] == 'Roi' and row[1] == 'dff']

        assert len(rows) == 1
        assert rows[0][2] == 'blob'
        assert rows[0][3] == 6
