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


class LinkError(RuntimeError):
    """Base for everything raised about links."""


class LinkTypeError(LinkError):
    """A link does not match what its kind is declared to connect."""


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
