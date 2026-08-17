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
