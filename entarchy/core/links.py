"""Link kinds: what they may connect, and in which direction.

A link kind is data, not a Python class, so that one can be invented at the
prompt the way an attribute name can. The cost of that choice is that nothing in
the code constrains what a kind connects, so the constraint has to be recorded
alongside the data and checked on every write. This module holds the part of
that check which needs no database.

An endpoint is described by an `Endpoint`: either an ordinary entity of a given
type, or a link of a given kind. The second form exists because every link
carries the same entity type, `LinkEntity`, so constraining a link endpoint by
entity type would allow connecting any two links at all - which is exactly the
confusion the registry is meant to prevent.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Union

# How many links of a kind are expected, which decides which guards apply when
#  they are written in bulk
CARDINALITIES = ('sparse', 'one_per_linker', 'dense')

DEFAULT_CARDINALITY = 'sparse'


# Writing more links than this in one call needs the count stated explicitly.
#  A runaway pairwise write is quiet: the truly enormous case dies building the
#  frame, but the middle of the range succeeds slowly and fills a disk.
MAX_LINKS_WITHOUT_CONFIRMATION = 100_000

# Fraction of the available pairs above which a 'sparse' kind is refused, and the
#  size below which density is not worth checking at all.
#
# Density alone is a bad signal: a link from one recording to each of its five
#  ROIs is 100% of the available pairs and entirely reasonable, and so is any
#  small cross product. What makes a dense write worth refusing is that the
#  per-link overhead becomes the dominant cost - roughly 2 kB against 4 bytes for
#  the same value in a matrix - and that only matters once there are many of them.
MAX_SPARSE_DENSITY = 0.5
MIN_COUNT_FOR_DENSITY_CHECK = 10_000

# Roughly what a link costs, measured at 1812 B on SQLite and 1956-2091 B on
#  MySQL with a few attributes. Used only to describe a refused or dry-run write.
APPROXIMATE_BYTES_PER_LINK = 2000


class LinkError(RuntimeError):
    """Base for everything raised about links."""


class LinkTypeError(LinkError):
    """A link does not match what its kind is declared to connect."""


class LinkDensityError(LinkError):
    """A write is dense enough that it should probably be a matrix."""


class LinkCardinalityError(LinkError):
    """A write would break the cardinality its kind declares."""


@dataclasses.dataclass(frozen=True)
class Endpoint:
    """What one end of a link kind may be.

    Exactly one of `entity_type` and `link_type` is set, or neither for a
    wildcard that accepts anything.
    """
    entity_type: Union[str, None] = None
    link_type: Union[str, None] = None

    def __post_init__(self):
        if self.entity_type is not None and self.link_type is not None:
            raise LinkTypeError('An endpoint is either an entity type or a link type, '
                                'not both.')

    @property
    def is_wildcard(self) -> bool:
        return self.entity_type is None and self.link_type is None

    @property
    def is_link(self) -> bool:
        return self.link_type is not None

    def accepts(self, other: 'Endpoint') -> bool:
        """Whether an actual endpoint satisfies this constraint."""
        if self.is_wildcard:
            return True
        if self.link_type is not None:
            return other.link_type == self.link_type
        return other.entity_type == self.entity_type

    def describe(self) -> str:
        if self.is_wildcard:
            return 'any'
        if self.link_type is not None:
            return f'{self.link_type} link'
        return self.entity_type

    def __str__(self):
        return self.describe()


@dataclasses.dataclass(frozen=True)
class LinkTypeSpec:
    """A link kind as the registry holds it."""
    name: str
    linker: Endpoint
    linked: Endpoint
    symmetric: bool = False
    cardinality: str = DEFAULT_CARDINALITY
    description: Union[str, None] = None

    def __post_init__(self):
        if self.cardinality not in CARDINALITIES:
            raise LinkTypeError(
                f'Unknown cardinality "{self.cardinality}" for link type "{self.name}". '
                f'Expected one of {", ".join(CARDINALITIES)}.')

        # Direction can only be inferred when the two ends differ. Where they are
        #  the same, symmetric or directed has to be stated, and a symmetric kind
        #  between different endpoint types makes no sense
        if self.symmetric and self.linker != self.linked:
            raise LinkTypeError(
                f'Link type "{self.name}" is symmetric, so both endpoints must be the '
                f'same; got {self.linker} and {self.linked}.')

    @property
    def endpoints_differ(self) -> bool:
        return self.linker != self.linked


def requires_direction_declaration(linker: Endpoint, linked: Endpoint) -> bool:
    """Whether the caller has to say symmetric or directed.

    Only when both ends are the same. Otherwise the endpoint types already say
    which end is which, and asking would be pedantry.
    """
    return linker == linked


def orientation(spec: LinkTypeSpec, linker: Endpoint, linked: Endpoint) -> str:
    """Check a pair against a kind and say how it should be stored.

    Returns 'as_given' or 'swapped'. Raises LinkTypeError if the pair does not
    match the kind in either order.

    Swapping is safe precisely when the declared endpoints differ: there is then
    only one way the pair can be meant, so rejecting `link(roi, phase, ...)` for
    a Phase -> Roi kind would be pedantry rather than safety.
    """
    if spec.linker.accepts(linker) and spec.linked.accepts(linked):
        return 'as_given'

    if spec.endpoints_differ and spec.linker.accepts(linked) and spec.linked.accepts(linker):
        return 'swapped'

    raise LinkTypeError(
        f'Link type "{spec.name}" connects {spec.linker} -> {spec.linked}, '
        f'but got {linker} -> {linked}.')


def canonical_pair(spec: LinkTypeSpec, linker_uuid: str, linked_uuid: str) -> tuple[str, str]:
    """Order the endpoints for storage.

    Symmetric kinds are stored smallest uuid first, so that a pair is found from
    either end without writing it twice. For those kinds the linker and linked
    labels carry no meaning afterwards, which is why queries against them may
    only address endpoints by type, never by role.
    """
    if spec.symmetric and linked_uuid < linker_uuid:
        return linked_uuid, linker_uuid

    return linker_uuid, linked_uuid


@dataclasses.dataclass
class LinkWriteResult:
    """What a bulk link write did, or would have done for a dry run."""
    link_type: str
    requested: int = 0
    created: int = 0
    already_present: int = 0
    duplicates_dropped: int = 0
    dry_run: bool = False
    link_uuids: list = dataclasses.field(default_factory=list)

    @property
    def estimated_bytes(self) -> int:
        return self.created * APPROXIMATE_BYTES_PER_LINK

    def __str__(self):
        verb = 'would create' if self.dry_run else 'created'
        parts = [f'{self.link_type}: {verb} {self.created} link(s) of {self.requested} '
                 f'requested']
        if self.already_present:
            parts.append(f'{self.already_present} already present')
        if self.duplicates_dropped:
            parts.append(f'{self.duplicates_dropped} duplicate(s) in the input')
        parts.append(f'~{self.estimated_bytes / 1024 ** 2:.1f} MB')

        return ', '.join(parts)


def check_write_size(spec: LinkTypeSpec, count: int, linker_count: int, linked_count: int,
                     confirm_count: int = None) -> None:
    """Refuse a write that looks like it forgot to be sparse.

    Two separate guards. The count ceiling catches sheer volume and is cleared by
    stating the number, so the caller has to have looked at it. The density check
    catches a large write that is most of every possible pair, where the two
    kilobytes an entity costs have swamped the four bytes of value it carries.

    Density is only consulted once the write is large, because on its own it says
    nothing: one recording linked to each of its five ROIs is 100% of the
    available pairs and perfectly reasonable.
    """
    if confirm_count is not None and confirm_count != count:
        raise LinkDensityError(
            f'confirm_count={confirm_count} does not match the {count} link(s) this '
            f'would write. Pass the actual number.')

    if spec.cardinality != 'dense' and count >= MIN_COUNT_FOR_DENSITY_CHECK:
        available = linker_count * linked_count
        if available > 0:
            density = count / available
            if density > MAX_SPARSE_DENSITY:
                raise LinkDensityError(
                    f'This would write {count} links of "{spec.name}" between '
                    f'{linker_count} and {linked_count} entities, which is '
                    f'{density:.0%} of every possible pair. At that density the link '
                    f'overhead dominates: about '
                    f'{count * APPROXIMATE_BYTES_PER_LINK / 1024 ** 2:.0f} MB as links '
                    f'against {available * 4 / 1024 ** 2:.1f} MB for the same values in '
                    f'a float32 matrix on the nearest common ancestor. Links buy '
                    f'queryability for that; if the trade is worth it here, declare the '
                    f'kind with cardinality="dense".')

    if confirm_count is None and count > MAX_LINKS_WITHOUT_CONFIRMATION:
        raise LinkDensityError(
            f'This would write {count} links of "{spec.name}", about '
            f'{count * APPROXIMATE_BYTES_PER_LINK / 1024 ** 3:.1f} GB. If that is '
            f'intended, pass confirm_count={count}.')


def resolve_endpoint(value: Any, entity_type_names: set, link_type_names: set) -> Endpoint:
    """Turn what a caller passed into an Endpoint.

    Accepts an Entity subclass, an entity type name, a registered link kind name,
    or None for a wildcard. Names are resolved against the entity types first,
    since those are fixed by the schema while link kinds come and go.
    """
    if value is None:
        return Endpoint()

    if isinstance(value, Endpoint):
        return value

    if isinstance(value, type):
        name = value.__name__
        if name not in entity_type_names:
            raise LinkTypeError(f'"{name}" is not an entity type of this entarchy.')
        return Endpoint(entity_type=name)

    if isinstance(value, str):
        if value in entity_type_names:
            return Endpoint(entity_type=value)
        if value in link_type_names:
            return Endpoint(link_type=value)

        raise LinkTypeError(
            f'"{value}" is neither an entity type of this entarchy nor a registered '
            f'link type. Entity types are fixed by the schema; a link type has to be '
            f'defined before it can be used as an endpoint.')

    raise LinkTypeError(f'Cannot use {value!r} as a link endpoint. Expected an entity '
                        f'class, an entity type name, a link type name, or None.')
