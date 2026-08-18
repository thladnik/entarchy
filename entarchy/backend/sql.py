"""Shared SQL storage backend.

Holds everything the SQLite and MySQL backends have in common: the table
definitions, blob serialization, query construction and every entity and
collection operation. Dialect specific behaviour is left to the subclasses
in sqlite.py and mysql.py through a small set of hooks.
"""
import contextlib
import hashlib
import io
import math
import operator
import os.path
import pathlib
import pickle
import re
import shutil
import datetime
import time
import warnings
from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import (Index, ForeignKey, LargeBinary, String, Text, create_engine,
                        BigInteger, Double)
from sqlalchemy.dialects.mysql import DATETIME as MYSQL_DATETIME, LONGBLOB, LONGTEXT
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from . import blob_store
from .backend import Backend
from .. import AnalysisEntity, Collection, Entarchy, Entity
from ..core.links import Endpoint, LinkTypeError, LinkTypeSpec


# MySQL's generic BLOB tops out at 64 KB, which is far too small for the
#  serialized arrays stored here, so that dialect gets LONGBLOB instead.
_BLOB_TYPE = LargeBinary().with_variant(LONGBLOB(), 'mysql')

# String attributes were VARCHAR(500). SQLite ignores a declared length, so a
#  longer string round-tripped there while MySQL raised "Data too long" on the
#  same write - the code worked or failed depending on the backend, and nothing
#  routes an oversized string anywhere else. MySQL's plain TEXT only moves the
#  boundary to 64 KB, so it gets LONGTEXT for the same reason as LONGBLOB above.
#  The column is not indexed, so there is no key length to stay under.
_TEXT_TYPE = Text().with_variant(LONGTEXT(), 'mysql')

# MySQL's DATETIME keeps whole seconds only and rounds on insert, so a row written
#  at 12:30:05.7 is stored as 12:30:06 - in the future relative to the moment it was
#  created. Collections filter on "created <= collection init time", so entities
#  added moments earlier would drop out of every query for up to a second.
#  Fractional seconds are requested explicitly; SQLite keeps full precision anyway.
_DATETIME_TYPE = sqlalchemy.DateTime().with_variant(MYSQL_DATETIME(fsp=6), 'mysql')


class Base(DeclarativeBase):
    pass


class EntityTypeTable(Base):
    __tablename__ = 'entity_types'

    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    parent_pk: Mapped[int] = mapped_column(ForeignKey('entity_types.pk'), nullable=True)
    parent: Mapped['EntityTypeTable'] = relationship('EntityTypeTable', back_populates='children', remote_side=[pk])
    children: Mapped[List['EntityTypeTable']] = relationship('EntityTypeTable', back_populates='parent', remote_side=[parent_pk])

    name: Mapped[str] = mapped_column(String(500), unique=True)

    entities: Mapped[List['EntityTable']] = relationship('EntityTable', back_populates='entity_type')


class EntityTable(Base):
    __tablename__ = 'entities'

    uuid: Mapped[str] = mapped_column(String(36), primary_key=True)
    parent_uuid: Mapped[str] = mapped_column(String(36), ForeignKey('entities.uuid'), nullable=True)
    entity_type_pk: Mapped[int] = mapped_column(ForeignKey('entity_types.pk'))

    id: Mapped[str] = mapped_column(String(500))

    # Many-to-One
    entity_type: Mapped['EntityTypeTable'] = relationship('EntityTypeTable', back_populates='entities')
    parent: Mapped['EntityTable'] = relationship('EntityTable', back_populates='children', remote_side=[uuid])

    # One-to-Many
    children: Mapped[List['EntityTable']] = relationship('EntityTable', back_populates='parent', remote_side=[parent_uuid])
    attributes: Mapped[List['AttributeTable']] = relationship('AttributeTable', back_populates='entity')

    created: Mapped[datetime.datetime] = mapped_column(_DATETIME_TYPE, default=datetime.datetime.now)
    modified: Mapped[datetime.datetime] = mapped_column(_DATETIME_TYPE, default=datetime.datetime.now,
                                                       onupdate=datetime.datetime.now)

    __table_args__ = (
        Index('ix_unique_id_per_parent_and_entity_type', 'parent_uuid', 'entity_type_pk', 'id', unique=True),
    )

    def __repr__(self):
        return f"<{self.entity_type.name}Row(id={self.id}, parent={self.parent})>"


class LinkTypeTable(Base):
    """The registry of link kinds.

    Kinds are data rather than Python classes, so that a link kind can be
    invented at the prompt the way an attribute name can. That means the
    database is the only registry there is, and the constraints on a kind have
    to live here too.
    """
    __tablename__ = 'link_types'

    name: Mapped[str] = mapped_column(String(255), primary_key=True)

    # Endpoint constraints. Per endpoint at most one of the two columns is set:
    #  an entity type for an ordinary entity, or a link type where the endpoint
    #  is itself a link. Both null is a deliberate wildcard.
    #  Links need the second form because every link carries the same entity type
    #  (LinkEntity), so constraining a link endpoint by entity type would permit
    #  connecting any two links at all.
    linker_type_pk: Mapped[int] = mapped_column(ForeignKey('entity_types.pk'), nullable=True)
    linker_link_type: Mapped[str] = mapped_column(String(255), ForeignKey('link_types.name'),
                                                  nullable=True)
    linked_type_pk: Mapped[int] = mapped_column(ForeignKey('entity_types.pk'), nullable=True)
    linked_link_type: Mapped[str] = mapped_column(String(255), ForeignKey('link_types.name'),
                                                  nullable=True)

    # Direction only has to be declared when both endpoints are the same kind;
    #  otherwise the endpoint types say which end is which
    symmetric: Mapped[bool] = mapped_column(default=False)

    # 'sparse' (the default), 'one_per_linker' or 'dense' - see core.links
    cardinality: Mapped[str] = mapped_column(String(50), default='sparse')

    description: Mapped[str] = mapped_column(String(2000), nullable=True)

    created: Mapped[datetime.datetime] = mapped_column(_DATETIME_TYPE, default=datetime.datetime.now)

    def __repr__(self):
        return f'<LinkType({self.name})>'


class Link(Base):
    """A relationship between two entities, carried by an entity of its own.

    link_uuid is the carrier entity's uuid, so a link's attributes go through
    exactly the same machinery as any other entity's. The carrier's parent is
    its linker, which keeps the entity tree valid and gives archive block files
    and map_async the same grouping they use everywhere else.
    """
    __tablename__ = 'links'

    link_uuid: Mapped[str] = mapped_column(String(36), ForeignKey('entities.uuid'),
                                           primary_key=True)
    link_type: Mapped[str] = mapped_column(String(255), ForeignKey('link_types.name'))

    linker_uuid: Mapped[str] = mapped_column(String(36), ForeignKey('entities.uuid'))
    linked_uuid: Mapped[str] = mapped_column(String(36), ForeignKey('entities.uuid'))

    entity = relationship('EntityTable', foreign_keys=[link_uuid])
    linker = relationship('EntityTable', foreign_keys=[linker_uuid])
    linked = relationship('EntityTable', foreign_keys=[linked_uuid])
    type = relationship('LinkTypeTable', foreign_keys=[link_type])

    created: Mapped[datetime.datetime] = mapped_column(_DATETIME_TYPE, default=datetime.datetime.now)
    modified: Mapped[datetime.datetime] = mapped_column(_DATETIME_TYPE, default=datetime.datetime.now,
                                                       onupdate=datetime.datetime.now)

    __table_args__ = (
        # One link of a given kind per ordered pair. The kind is part of the key
        #  so that the same pair can carry several kinds at once - a phase and a
        #  ROI may have both a mean_response and a peak_latency between them.
        Index('ix_unique_link_per_type_and_pair',
              'link_type', 'linker_uuid', 'linked_uuid', unique=True),
        # Finding a link from its linked end, which a plain (linker, linked)
        #  index cannot serve
        Index('ix_link_reverse', 'link_type', 'linked_uuid'),
        Index('ix_link_linker', 'linker_uuid'),
        Index('ix_link_linked', 'linked_uuid'),
    )

    def __repr__(self):
        return f'<Link({self.link_type}: {self.linker_uuid} -> {self.linked_uuid})>'


class AttributeTable(Base):
    __tablename__ = 'attributes'

    entity_uuid: Mapped[str] = mapped_column(String(36), ForeignKey('entities.uuid'), primary_key=True)
    entity: Mapped['EntityTable'] = relationship('EntityTable', foreign_keys=[entity_uuid], back_populates='attributes')
    analysis_uuid: Mapped[str] = mapped_column(String(36), nullable=True)

    name: Mapped[str] = mapped_column(String(500), primary_key=True, index=True)

    value_str: Mapped[str] = mapped_column(_TEXT_TYPE, nullable=True)
    value_int: Mapped[int] = mapped_column(BigInteger(), nullable=True)
    value_float: Mapped[float] = mapped_column(Double(), nullable=True)
    value_bool: Mapped[bool] = mapped_column(nullable=True)
    value_date: Mapped[datetime.date] = mapped_column(nullable=True)
    value_datetime: Mapped[datetime.datetime] = mapped_column(_DATETIME_TYPE, nullable=True)
    value_blob: Mapped[bytes] = mapped_column(_BLOB_TYPE, nullable=True)
    data_type: Mapped[str] = mapped_column(String(500), nullable=True)
    data_size: Mapped[int] = mapped_column(BigInteger(), nullable=False, default=0)

    float_is_nan: Mapped[bool] = mapped_column(default=False)
    float_is_inf: Mapped[bool] = mapped_column(default=False)

    mutable = mapped_column(sqlalchemy.Boolean, default=True)

    created: Mapped[datetime.datetime] = mapped_column(_DATETIME_TYPE, default=datetime.datetime.now)
    modified: Mapped[datetime.datetime] = mapped_column(_DATETIME_TYPE, default=datetime.datetime.now,
                                                       onupdate=datetime.datetime.now)

    # (entity_uuid, name) is already the primary key, which every dialect backs
    #  with a unique index of its own - SQLite as sqlite_autoindex_attributes_1,
    #  InnoDB as the clustered index. A second unique index on the same two
    #  columns in the same order was declared here and simply duplicated it:
    #  ANALYZE reported identical statistics for both, and it cost 44 MB on a
    #  27 000 ROI entarchy (65 bytes per attribute row) plus its share of every
    #  insert. Dropped.
    #
    #  The single-column index on `name` (declared on the column itself) is a
    #  different matter and load bearing: it answers "which entities have this
    #  attribute", which is what every ../parent and [Ancestor] filter needs.

    def __repr__(self):
        return f"<Attribute({self.name}, {self.entity})>"


def _retry_on_operational_failure(fun: Callable, retry_num: int = 3) -> Callable:
    """Decorator for catching sqlalchemy.exc.OperationalError,
    which is typically emitted when a connection has been disconnected by the host.
    """

    assert retry_num > 0, 'Need at least one retry'

    def _wrapper(self, *args, **kwargs):
        i = 0
        err = None
        while i <= retry_num:

            try:
                return fun(self, *args, **kwargs)

            except OperationalError as e:

                # Inside a batch the rollback below would discard every write the
                #  batch has already made, and retrying only this call would not
                #  bring them back. Let the error out so the whole batch rolls
                #  back as a unit and the caller can retry it.
                if getattr(self, '_batch_depth', 0) > 0:
                    raise

                err = e
                i += 1

                # Roll the session back, otherwise the retry reuses a session
                #  that is still in a failed transaction state
                try:
                    self.sql_session.rollback()
                except Exception:
                    pass

                time.sleep(min(2 ** i, 5))

        if err is not None:
            raise err

        return None

    return _wrapper


class _QueryContext:
    """What a filter expression is being resolved against.

    For an ordinary collection that is just the entity type. For a collection of
    links it is also the link kind, which is what lets `@Roi.attr` work out which
    endpoint is meant.
    """

    def __init__(self, entity_type_name: str, link_spec: LinkTypeSpec = None):
        self.entity_type_name = entity_type_name
        self.link_spec = link_spec


def _build_query_from_collection(coll: Collection,
                                 sess: sqlalchemy.orm.Session
                                 ) -> sqlalchemy.orm.Query:

    entity_type_name = coll.entity_type.__name__
    as_tree = coll.as_tree
    creation_time = coll.init_time

    # Create base query
    _query = sess.query(EntityTable).join(EntityTypeTable).filter(EntityTypeTable.name == entity_type_name)
    _query = _query.filter(EntityTable.created <= creation_time)

    link_spec = None
    if getattr(coll, 'link_type', None) is not None:
        # A link collection is a collection of carrier entities narrowed to one kind
        link_spec = coll.link_type_spec
        _query = (_query.join(Link, Link.link_uuid == EntityTable.uuid)
                  .filter(Link.link_type == coll.link_type))

        endpoint_filter = _endpoint_membership_filter(coll, sess)
        if endpoint_filter is not None:
            _query = _query.filter(endpoint_filter)

    # Apply filters generated from the abstract syntax tree
    if len(as_tree) == 0:
        return _query

    filters = _generate_attribute_filters(
        _QueryContext(entity_type_name, link_spec), sess, as_tree)

    return _query.filter(filters)


def _endpoint_membership_filter(coll: Collection, sess: sqlalchemy.orm.Session):
    """Restrict a link collection to links whose ends lie in given collections.

    Each constraint is a (linker collection, linked collection) pair, either side
    None for "anything"; the pairs are OR-ed. Membership becomes a subquery over
    the member collection's own query, so it carries that collection's filters
    and is not limited by how many parameters a statement may bind - spelling the
    members out as uuids gives up around sixteen thousand of them.
    """
    constraints = getattr(coll, 'endpoint_constraints', None)
    if not constraints:
        return None

    clauses = []
    for linker_collection, linked_collection in constraints:
        parts = []
        if linker_collection is not None:
            parts.append(Link.linker_uuid.in_(
                _uuids_of(_build_query_from_collection(linker_collection, sess))))
        if linked_collection is not None:
            parts.append(Link.linked_uuid.in_(
                _uuids_of(_build_query_from_collection(linked_collection, sess))))

        if len(parts) == 1:
            clauses.append(parts[0])
        elif len(parts) > 1:
            clauses.append(sqlalchemy.and_(*parts))

    if len(clauses) == 0:
        return None

    return clauses[0] if len(clauses) == 1 else sqlalchemy.or_(*clauses)


# Endpoint roles that address a link's ends rather than an entity type
_ENDPOINT_ROLES = ('linker', 'linked', 'either', 'both')


def _split_endpoint_name(name: str) -> tuple[str, str]:
    """Split "@role.rest" into its role and the attribute path after it."""
    role, _, rest = name[1:].partition('.')

    if not role or not rest:
        raise ValueError(f'Malformed endpoint reference "{name}". Expected something '
                         f'like "@Roi.attribute" or "@linker.attribute".')

    return role, rest


def _resolve_endpoint_roles(context: _QueryContext, role: str, name: str) -> list[str]:
    """Which end(s) of the link "@role" addresses."""
    spec = context.link_spec

    if spec is None:
        raise ValueError(
            f'"{name}" addresses a link endpoint, but this is a collection of '
            f'{context.entity_type_name} rather than of links. Endpoint filters are '
            f'only available on a link collection (entarchy.links(kind, ...)).')

    if role in ('linker', 'linked'):
        if spec.symmetric:
            raise ValueError(
                f'"{name}" addresses the {role} of "{spec.name}", which is symmetric: '
                f'its two ends are stored in uuid order, so which is which carries no '
                f'meaning. Use @either or @both, or address the endpoint by type.')
        return [role]

    if role == 'either':
        return ['linker', 'linked']
    if role == 'both':
        return ['linker', 'linked']

    # Otherwise the role names an entity type or link kind; find which end it is
    matches = [end for end, endpoint in (('linker', spec.linker), ('linked', spec.linked))
               if endpoint.entity_type == role or endpoint.link_type == role]

    if len(matches) == 0:
        raise ValueError(
            f'"{name}" addresses a "{role}" endpoint, but "{spec.name}" connects '
            f'{spec.linker} to {spec.linked}. Use one of those, or @linker/@linked/'
            f'@either/@both.')

    if len(matches) == 2 and not spec.symmetric:
        raise ValueError(
            f'"{name}" is ambiguous: both ends of "{spec.name}" are {role}. Use '
            f'@linker or @linked to say which, or @either/@both for either or both.')

    return matches


def _endpoint_attribute_filter(context: _QueryContext, _session, name: str, comparison):
    """Build the filter for an "@endpoint.attribute" reference.

    Returns a filter over EntityTable.uuid rather than a subquery, so that
    "either" and "both" become OR and AND of two membership tests. Expressing
    them as UNION and INTERSECT instead would work on SQLite but INTERSECT is
    only available on recent MySQL.
    """
    role, rest = _split_endpoint_name(name)
    ends = _resolve_endpoint_roles(context, role, name)
    spec = context.link_spec

    filters = []
    for end in ends:
        endpoint_column = Link.linker_uuid if end == 'linker' else Link.linked_uuid
        endpoint = spec.linker if end == 'linker' else spec.linked

        parent_level, attr_name = _split_endpoint_traversal(_session, rest, endpoint, name)

        # Walk from the endpoint entity up to whichever ancestor holds the attribute
        aliases = [sqlalchemy.orm.aliased(EntityTable) for _ in range(parent_level + 1)]
        subq = (_session.query(Link.link_uuid.label('entity_uuid'))
                .filter(Link.link_type == spec.name)
                .join(aliases[0], aliases[0].uuid == endpoint_column))

        for level in range(parent_level):
            subq = subq.join(aliases[level + 1],
                             aliases[level].parent_uuid == aliases[level + 1].uuid)

        anchor = aliases[-1]
        subq = subq.join(AttributeTable, AttributeTable.entity_uuid == anchor.uuid)
        subq = subq.filter(AttributeTable.name == attr_name)
        if comparison is not None:
            subq = subq.filter(comparison)

        filters.append(EntityTable.uuid.in_(_session.query(subq.subquery().c.entity_uuid)))

    if len(filters) == 1:
        return filters[0]

    return sqlalchemy.and_(*filters) if role == 'both' else sqlalchemy.or_(*filters)


def _split_endpoint_traversal(_session, rest: str, endpoint: Endpoint,
                              name: str) -> tuple[int, str]:
    """How far above the endpoint the attribute sits, and its name."""
    if rest.startswith('../'):
        parent_level = rest.count('../')
        return parent_level, rest[parent_level * 3:]

    if rest.startswith('['):
        if ']' not in rest:
            raise ValueError(f'Malformed ancestor reference in "{name}".')

        ancestor_type, attr_name = rest.replace('[', '').split(']')

        if endpoint.entity_type is None:
            raise ValueError(
                f'"{name}" walks up from an endpoint declared as {endpoint}, whose '
                f'entity type is not fixed, so "{ancestor_type}" cannot be resolved.')

        parent_level = _get_entity_type_ancestor_distance(_session, endpoint.entity_type,
                                                          ancestor_type)
        if parent_level is None:
            raise ValueError(f'Entity type "{ancestor_type}" is not an ancestor of '
                             f'"{endpoint.entity_type}" in "{name}".')

        return parent_level, attr_name

    return 0, rest


def _generate_attribute_filters(context: _QueryContext,
                                _session: sqlalchemy.orm.Session,
                                as_tree: dict[str, ...]) -> Any:

    entity_type_name = context.entity_type_name

    _operator = as_tree['operator'].upper()

    # Handle connectives
    if _operator in ('AND', 'OR', 'XOR'):
        left = _generate_attribute_filters(context, _session, as_tree['left_operand'])
        right = _generate_attribute_filters(context, _session, as_tree['right_operand'])

        if _operator == 'AND':
            return sqlalchemy.and_(left, right)
        elif _operator == 'OR':
            return sqlalchemy.or_(left, right)
        # XOR: (left OR right) AND NOT (left AND right)
        return sqlalchemy.and_(sqlalchemy.or_(left, right),
                               sqlalchemy.not_(sqlalchemy.and_(left, right)))

    # Handle comparisons
    elif _operator in ('IN', '<=', '<', '==', '!=', '>', '>='):

        name = as_tree['left_operand']
        value = as_tree['right_operand']

        if _operator == 'IN':
            if not isinstance(value, list) or len(value) == 0:
                raise ValueError('Operand after IN statement should be a non-empty list of values')

            columns = _value_columns(value[0])
            comparisons = [column.in_(value) for column in columns]
        else:
            columns = _value_columns(value)

            _op_fun = {
                '<': operator.lt,
                '<=': operator.le,
                '==': operator.eq,
                '!=': operator.ne,
                '>=': operator.ge,
                '>': operator.gt
            }[_operator]

            comparisons = [_op_fun(column, value) for column in columns]

        comparison = (comparisons[0] if len(comparisons) == 1
                      else sqlalchemy.or_(*comparisons))

        # An @ prefix addresses one of a link's endpoints
        if name.startswith('@'):
            return _endpoint_attribute_filter(context, _session, name, comparison)

        # If name does not start with dots, it's a direct attribute
        if not name.startswith('../') and not name.startswith('['):
            # Build the subquery to filter entities matching the comparison
            subquery = (_session.query(AttributeTable.entity_uuid)
                        .filter(AttributeTable.name == name, comparison)
                        .join(EntityTable)
                        .join(EntityTypeTable).filter(EntityTypeTable.name == entity_type_name)
                        .subquery())

        # If name starts with '../', it's a parent attribute, each '../' indicates one level up
        # Alternatively, explicit parent entity type names may be used: [ParentEntityTypeName]attribute_name
        else:

            if name.startswith('../'):
                parent_level = name.count('../')
                attr_name = name[parent_level * 3:]  # remove leading "../" occurrences

            else:
                if not (('[' in name) and (']' in name)):
                    raise ValueError('Malformed attribute name for parent entity traversal.')
                parent_entity_type_name, attr_name = name.replace('[', '').split(']')
                parent_level = _get_entity_type_ancestor_distance(_session, entity_type_name, parent_entity_type_name)

                if parent_level is None:
                    raise ValueError(f'Entity type "{parent_entity_type_name}" is not an ancestor of "{entity_type_name}".')
                # print(f'Going {parent_level} up from {entity_type_name} to {parent_entity_type_name}')

            if not attr_name:
                raise ValueError('Attribute name after ../ traversal is empty.')

            # Create aliases
            entity_aliases = [sqlalchemy.orm.aliased(EntityTable) for _ in range(parent_level + 1)]

            # Start query selecting the current entity uuid
            subq = _session.query(entity_aliases[0].uuid.label('entity_uuid'))

            # Join parent chain
            for i in range(parent_level):
                left = entity_aliases[i]
                right = entity_aliases[i + 1]
                subq = subq.join(right, left.parent_uuid == right.uuid)

            # Join parent attributes and apply attribute name and comparison there
            ancestor = entity_aliases[-1]
            subq = subq.join(AttributeTable, AttributeTable.entity_uuid == ancestor.uuid)
            subq = subq.filter(AttributeTable.name == attr_name, comparison)

            # Ensure current entity type matches the requested collection entity_type_name
            subq = subq.join(EntityTypeTable, entity_aliases[0].entity_type_pk == EntityTypeTable.pk)
            subq = subq.filter(EntityTypeTable.name == entity_type_name)

            subquery = subq.subquery()

    # Handle unary operators
    elif _operator == 'EXIST':
        name = as_tree['right_operand']

        if name.startswith('@'):
            return _endpoint_attribute_filter(context, _session, name, None)

        subquery = (_session.query(AttributeTable.entity_uuid)
                    .filter(AttributeTable.name == name)
                    .subquery())

    elif _operator == 'NOT':
        return sqlalchemy.not_(_generate_attribute_filters(context, _session,
                                                           as_tree['right_operand']))

    # Fallback
    else:
        print(f'Unknown unary operator: {_operator}', as_tree)
        raise ValueError('Unexpected operator in the expression tree')

    # Return the `IN` filter to apply to the main query
    return EntityTable.uuid.in_(_session.query(subquery.c.entity_uuid))


def _get_entity_type_ancestor_distance(session: sqlalchemy.orm.Session,
                                       entity_type_name: str,
                                       parent_entity_type_name: str) -> int | None:
    """
    Return number of parent steps required to reach `parent_entity_type_name`
    starting from `entity_type_name`. Return 0 if names are equal, or None
    if `parent_entity_type_name` is not an ancestor.
    """
    # Quick equality check
    if entity_type_name == parent_entity_type_name:
        return 0

    row = session.query(EntityTypeTable).filter(EntityTypeTable.name == entity_type_name).one_or_none()
    if row is None:
        return None

    distance = 0
    visited = set()
    current = row

    while current is not None:
        # protect against cycles
        if current.pk in visited:
            return None
        visited.add(current.pk)

        # move to parent
        parent = current.parent
        distance += 1
        if parent is None:
            return None

        if parent.name == parent_entity_type_name:
            return distance

        current = parent

    return None


# SQLite refuses any statement carrying more than SQLITE_MAX_VARIABLE_NUMBER bound
#  parameters - 32766 since 3.32, and 999 before that. The attribute upsert binds one
#  parameter per column per row, so a single statement covering a large collection
#  goes over: at the width of the attributes table that is about 1900 entities, which
#  is well inside the range entarchy is built for. MySQL has no equivalent limit, since
#  PyMySQL interpolates literals rather than binding, but bounding the statement size
#  keeps it clear of max_allowed_packet as well.
MAX_BOUND_PARAMETERS = 30_000


def _chunk_by_bound_parameters(records: list[dict], columns: int):
    """Split rows so that no single statement exceeds the bound parameter limit."""
    if len(records) == 0:
        return

    rows_per_statement = max(1, MAX_BOUND_PARAMETERS // max(1, columns))
    for start in range(0, len(records), rows_per_statement):
        yield records[start:start + rows_per_statement]


# The stored types a minimum, a maximum and a distinct count mean something
#  for. A blob has none of the three, and asking would mean decoding it.
_DISTRIBUTION_TYPES = ('str', 'int', 'float', 'bool', 'date', 'datetime')


def _uuids_of(entity_query: sqlalchemy.orm.Query):
    """A subquery selecting a collection's entity uuids, for use with IN.

    Passing `query.subquery().primary_key` to `in_()` looks right but is not: it
    is a collection of Column objects, so SQLAlchemy renders the subquery into
    the FROM clause and compares against its column - a cartesian product, with
    a membership test per combined row:

        FROM attributes JOIN entities ON ..., (SELECT ...) AS anon_1
        WHERE entities.uuid IN (anon_1.uuid)

    With 713 000 attribute rows and 27 000 entities that is 1.9e10 combinations,
    which never finished. Small fixtures hid it completely.
    """
    subquery = entity_query.subquery()

    return sqlalchemy.select(subquery.c.uuid)


def _value_columns(value: Any) -> list:
    """The attribute columns a filter literal could match.

    A numeric literal has to consider both the integer and the float column.
    Which column an attribute lands in is decided by how the value was written,
    not by how a filter spells its threshold, so "depth > 0" must still find a
    depth of 15.0 and "npix > 100.0" an npix of 153. Matching on the literal's
    type alone silently returned nothing in those cases, which is worse than an
    error because an empty collection looks like an answer.

    bool is checked by class name rather than isinstance, since in Python it is
    a subclass of int and must keep going to value_bool.
    """
    type_name = value.__class__.__name__

    if type_name in ('int', 'float'):
        return [AttributeTable.value_int, AttributeTable.value_float]

    return [getattr(AttributeTable, f'value_{type_name}')]


def _get_namehash(name: str) -> str:
    return hashlib.sha224(name.encode()).hexdigest()


# def _get_attribute_fp(_entity: Entity, row: AttributeTable, _format) -> tuple[str, str]:
#
#     _uuid = _entity.uuid.replace('-', '')
#     _shards = [_uuid[4*i:4*(i+1)] for i in range(8)]  # Create path shards from uuid (8x4 characters for uuid4)
#     fp = os.path.join(_entity.entarchy.path, 'ext', *_shards)
#
#     return pathlib.Path(fp).as_posix(), f'{_get_namehash(row.name)}.{_format}'

def _get_attribute_fp(_ent_path: str, _entity_uuid: str, attr_name: str, _format) -> tuple[str, str]:

    _uuid = _entity_uuid.replace('-', '')
    _shards = [_uuid[4*i:4*(i+1)] for i in range(8)]  # Create path shards from uuid (8x4 characters for uuid4)
    fp = os.path.join(_ent_path, 'ext', *_shards)

    return pathlib.Path(fp).as_posix(), f'{_get_namehash(attr_name)}.{_format}'


def _read_attribute_data(_entity: Entity, row: AttributeTable):
    if row.data_type is None:
        raise ValueError('Attribute data type is None.')

    # Load blob
    if row.data_type == 'blob':
        entarchy = _entity.entarchy
        try:
            return blob_store.loads(row.value_blob, root_path=entarchy.path,
                                    memmap=getattr(entarchy.backend, 'memmap', False))
        except Exception as e:
            store = '<unreadable>'
            try:
                store = blob_store.store_of(row.value_blob)
            except Exception:
                pass

            print(f'Failed to read blob attribute "{row.name}" of entity "{_entity}".')
            print(f'Stored in: {store}')
            print('This may be caused by a missing file on disk if the value was stored '
                  'outside the database, or by corrupted data in the row.')

            raise e

    # Otherwise load from this row based on data type
    val = getattr(row, f'value_{row.data_type}')

    # Check for inf and NaNs
    if row.data_type == 'float' and val is None:
        if row.float_is_inf:
            # Sign of infinity is kept in the (otherwise unused) value_int column
            return -1.0 * float('inf') if row.value_int < 0 else float('inf')
        elif row.float_is_nan:
            return float('nan')

    return val


def _write_attribute_data(_entity: Entity, row: AttributeTable, data: Any):

    # TODO: in future version, information about data type byte number should be included in data_type column
    #  This way the exact data type can be restored upon read (e.g. int8, int16, float32, float64, etc.)
    #  This would mean that python native scalars may be stored as regular 64bit,
    #  while numpy scalars get variable sizes.
    #  This won't affect actual storage though, as the database will use the same column types (bigint, double) anyway.

    # Get corresponding builtin python scalar type for numpy scalars
    if isinstance(data, np.generic):
        data = data.item()

    # If previous data type was float, reset flags
    if row.data_type == 'float':
        row.float_is_inf = False
        row.float_is_nan = False
        row.value_int = None  # may hold the sign of a previous inf value

    # A value the row kept outside the database, which may be left behind by
    #  what is written below
    root_path = _entity.entarchy.path
    replaced_file = (_owned_file_of(row.value_blob, root_path)
                     if row.data_type == 'blob' else None)

    # Set (potential) previous value to None
    if row.data_type is not None:
        row.__setattr__(f'value_{row.data_type}', None)

    # Handle scalars and datetime values
    if type(data) in (str, float, int, bool, datetime.date, datetime.datetime):

        # Set value type
        data_type_map = {str: 'str', float: 'float', int: 'int',
                         bool: 'bool', datetime.date: 'date', datetime.datetime: 'datetime'}
        data_type = data_type_map.get(type(data))

        # Some SQL dialects don't support inf float values
        if data_type == 'float' and math.isinf(data):
            row.float_is_inf = True
            # Preserve the sign of infinity in the otherwise unused value_int column
            row.value_int = 1 if data > 0 else -1
            data = None
        elif data_type == 'float' and math.isnan(data):
            row.float_is_nan = True
            data = None

        if data_type in ('int', 'float'):
            data_size = 8
        elif data_type == 'bool':
            data_size = 1
        elif data_type == 'date':
            data_size = 3
        elif data_type == 'datetime':
            data_size = 8  # depends on fractional seconds, but max. 8
        elif data_type == 'str':
            data_size = len(data.encode('utf-8'))
        else:
            raise RuntimeError(f'Unsupported data type {data_type}.')

    # Set value on corresponding column based on type
    else:
        data_type = 'blob'
        data = _store_blob(data, root_path,
                           _entity.entarchy.max_blob_size, _entity.uuid, row.name)
        # For a value kept outside the row, the size that matters is the file's
        data_size = _stored_size(data, root_path)

    # Write to row
    row.data_type = data_type
    row.data_size = data_size
    # Set data
    row.__setattr__(f'value_{data_type}', data)

    _discard_replaced_file(replaced_file, data if data_type == 'blob' else None,
                           root_path)


def _store_blob(value: Any, root_path: str, max_blob_size: int,
                entity_uuid: str, attr_name: str) -> bytes:
    """Encode a value and decide whether it goes in the row or in a file.

    Values at or above max_blob_size are written to `ext/`, and the row keeps a
    pointer. The file holds the same encoded bytes, so both paths read alike.
    """
    if isinstance(value, blob_store.MediaFile):
        return _store_media(value, root_path, entity_uuid, attr_name)

    encoded = blob_store.dumps(value, where=attr_name)

    if len(encoded) < max_blob_size:
        return encoded

    directory, filename = _get_attribute_fp(root_path, entity_uuid, attr_name, 'blob')
    os.makedirs(directory, exist_ok=True)

    full_path = f'{directory}/{filename}'
    with open(full_path, 'wb') as f:
        f.write(encoded)

    return blob_store.dumps_external(
        pathlib.Path(full_path).relative_to(root_path).as_posix())


# Two shard levels rather than the eight `ext/` uses. Sharding by uuid puts one
#  directory per entity either way - the extra levels spread nothing, since the
#  leaf holds only that entity's files - and every level costs path length that
#  Windows counts against a 260 character limit.
MEDIA_SHARD_LEVELS = 2


def _get_media_fp(root_path: str, entity_uuid: str, attr_name: str,
                  source_name: str) -> tuple[str, str]:
    """Where a media file goes: sharded by entity, named for its attribute.

    Named the way `ext/` payloads are - the hash of the attribute name - rather
    than after the source file. An acquisition file name can be seventy
    characters of instrument settings, and putting it in the path makes the
    length of every media path depend on whatever the microscope wrote.

    The extension is kept, because it is short, it tells a reader browsing the
    directory what kind of file this is, and some decoders dispatch on it.
    """
    _uuid = entity_uuid.replace('-', '')
    shards = [_uuid[4 * i:4 * (i + 1)] for i in range(MEDIA_SHARD_LEVELS)]
    directory = pathlib.PurePath(root_path, 'media', *shards).as_posix()

    extension = re.sub(r'[^A-Za-z0-9.]+', '', os.path.splitext(source_name)[1])[:16]

    return directory, f'{_get_namehash(attr_name)}{extension}'


def _store_media(media: 'blob_store.MediaFile', root_path: str,
                 entity_uuid: str, attr_name: str) -> bytes:
    """Take a media file into the entarchy and return the pointer to it.

    An entarchy is meant to be self-contained, so the file is copied in rather
    than referenced where it lies. A MediaFile that is already stored under this
    root is copied to the place this attribute would put it, so that two
    entities never share one file - there is no reference counting, and deleting
    one attribute would otherwise take the other's data with it.
    """
    source = media.path
    if not os.path.exists(source):
        raise FileNotFoundError(f'Cannot store media attribute "{attr_name}": '
                                f'"{source}" does not exist.')

    directory, filename = _get_media_fp(root_path, entity_uuid, attr_name,
                                        media.relative_path or source)
    os.makedirs(directory, exist_ok=True)
    destination = f'{directory}/{filename}'

    if os.path.abspath(source) != os.path.abspath(destination):
        if media.move:
            shutil.move(source, destination)
        else:
            shutil.copyfile(source, destination)

    return blob_store.dumps_media({
        'path': pathlib.Path(destination).relative_to(root_path).as_posix(),
        'media_type': media.media_type,
        'bytes': os.path.getsize(destination),
        'sha256': blob_store.sha256_of(destination),
    })


def _stored_size(raw: bytes, root_path: str) -> int:
    """How many bytes this value actually occupies, wherever they are.

    A pointer is a few hundred bytes; the file it names can be gigabytes, and
    data_size is meant to say where the storage went.
    """
    owned = _owned_file_of(raw, root_path)
    if owned is None:
        return len(raw)

    try:
        return os.path.getsize(owned)
    except OSError:
        return len(raw)


def _owned_file_of(raw: bytes, root_path: str) -> Union[str, None]:
    """The file a stored value keeps outside the row, if it keeps one."""
    if raw is None:
        return None

    try:
        header, _ = blob_store._unpack(raw)
    except ValueError:
        return None

    relative = header.get('x') or (header['m']['path'] if 'm' in header else None)

    return None if relative is None else os.path.join(root_path, relative)


def _discard_replaced_file(previous: Union[str, None], written: bytes,
                           root_path: str) -> None:
    """Remove the file a replaced value left behind.

    An `ext/` payload keeps the same name when it is rewritten, so it is
    overwritten in place - but a value that shrinks below max_blob_size, or a
    media file replaced by one with a different name, would otherwise leave the
    old file on disk for good.
    """
    if previous is None:
        return

    current = _owned_file_of(written, root_path)
    if current is not None and os.path.abspath(current) == os.path.abspath(previous):
        return

    try:
        os.remove(previous)
    except OSError:
        # Never fail a write because a stale file could not be cleaned up
        pass


def _deserialize(data: bytes, _entarchy: Entarchy) -> Any:

    return blob_store.loads(data, root_path=_entarchy.path,
                            memmap=getattr(_entarchy.backend, 'memmap', False))


class SQLBackend(Backend):
    _sql_engine: sqlalchemy.Engine | None = None
    _sql_session: sqlalchemy.orm.Session | None = None
    _db_triggers_enabled: bool = None
    _batch_depth: int = 0

    @contextlib.contextmanager
    def batch(self):
        """Collect the writes inside this block into a single transaction.

        A commit costs an fsync, and that dominates everything else when many
        entities are written one after another: measured at 5.91 ms for an insert
        plus commit against 0.028 ms for the same insert inside a batch. Writing
        a thousand entities therefore spent most of its time waiting on the disk
        rather than doing work.

        It also makes `with ent:` mean what it appears to mean. Committing per
        entity left a failure halfway through a block with half the entities
        already persisted; a batch either lands completely or not at all.
        """
        self._batch_depth += 1
        try:
            yield
        except BaseException:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                self.sql_session.rollback()
                self.sql_session.close()
            raise

        self._batch_depth -= 1
        if self._batch_depth == 0:
            self.sql_session.commit()
            self.sql_session.close()

    @contextlib.contextmanager
    def _write_session(self):
        """Session for a write, left open across calls while a batch is running."""
        if self._batch_depth > 0:
            yield self.sql_session
        else:
            with self.sql_session as session:
                yield session

    def _commit(self, session: sqlalchemy.orm.Session) -> None:
        """Commit, unless a batch is collecting writes into one transaction."""
        if self._batch_depth > 0:
            # Flush so later statements in the batch see these rows, but leave
            #  the transaction open for the single commit at the end
            session.flush()
        else:
            session.commit()

    @property
    def dbname(self) -> str:
        return self._config['dbname']

    # Dialect specific hooks. Subclasses implement the handful of things that
    #  genuinely differ between SQLite and MySQL; everything else is shared.

    def _create_engine(self) -> sqlalchemy.Engine:
        """Build the SQLAlchemy engine for this backend."""
        raise NotImplementedError

    def _create_database(self) -> None:
        """Create the database or schema itself, if the dialect needs it."""

    def _create_triggers(self) -> None:
        """Install triggers that maintain entity modification times, if supported."""

    def _drop_storage(self) -> None:
        """Remove the database or schema. Connections are already closed."""
        raise NotImplementedError

    def _insert_statement(self, values: list[dict]):
        """Dialect specific INSERT for the attributes table."""
        raise NotImplementedError

    def _inserted_values(self, insert_statement):
        """The pseudo-table holding the values proposed by an insert."""
        raise NotImplementedError

    def _upsert_statement(self, insert_statement, update_values: dict):
        """Turn an insert into an upsert on the attributes primary key."""
        raise NotImplementedError

    @property
    def sql_engine(self) -> sqlalchemy.Engine:
        if not hasattr(self, '_sql_engine') or self._sql_engine is None:
            self._sql_engine = self._create_engine()

        return self._sql_engine

    @property
    def sql_session(self) -> sqlalchemy.orm.Session:
        if not hasattr(self, '_sql_session') or self._sql_session is None:
            self._sql_session = sqlalchemy.orm.Session(self.sql_engine)
            
        return self._sql_session

    @Backend.debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value
        if self._sql_engine is not None:
            self._sql_engine.echo = value

    @property
    def db_triggers_enabled(self) -> bool:
        """Whether the database maintains entity modification times itself."""
        if self._db_triggers_enabled is None:
            self._db_triggers_enabled = False

        return self._db_triggers_enabled

    @_retry_on_operational_failure
    def create(self) -> bool:

        self._create_database()

        print('> Create tables')
        Base.metadata.create_all(self.sql_engine)

        self._create_triggers()

        return True

    @_retry_on_operational_failure
    def create_type_hierarchy(self, _hierarchy: dict[str, ...]) -> bool:

        with self.sql_session as session:
            def _create_entity_type(_hierarchy: dict[str, ...], parent_row: Union[EntityTypeTable, None]):
                for name, children in _hierarchy.items():
                    row = EntityTypeTable(name=name, parent=parent_row)
                    session.add(row)
                    _create_entity_type(children, row)

            # Add custom types
            _create_entity_type(_hierarchy, None)
            session.commit()

        return True

    @_retry_on_operational_failure
    def delete(self, confirm: bool = False):

        if not confirm:
            raise RuntimeError('Failed to delete backend. Confirmation not provided.')

        # Release handles first; an open database cannot be removed on Windows
        self.close()

        self._drop_storage()

    # Entity related methods

    @_retry_on_operational_failure
    def add_entities(self, _entities: list[Entity]) -> bool:

        with self._write_session() as session:

            # Get entity type map
            entity_type_map = {None: None}
            for row in session.query(EntityTypeTable).all():
                entity_type_map[row.name] = row

            # Create entity rows
            new_entity_rows = []
            for _entity in _entities:
                row = EntityTable(
                    uuid=_entity.uuid,
                    id=_entity.id,
                    parent_uuid=_entity.parent.uuid if _entity.parent is not None else None,
                    entity_type=entity_type_map[_entity.__class__.__name__]
                )
                new_entity_rows.append(row)

            # Add and commit
            session.add_all(new_entity_rows)
            self._commit(session)

        return True

    def get_entity_attribute(self, _entity: Entity, name: str) -> Any:

        return self.get_entity_attributes(_entity, [name])[0]

    @_retry_on_operational_failure
    def get_entity_attribute_names(self, _entity: Entity) -> list[str]:

        entity_uuid = _entity.uuid

        with self.sql_session as session:

            query = session.query(AttributeTable.name).filter(AttributeTable.entity_uuid == entity_uuid)

            names = [row.name for row in query.all()]

        return names

    @_retry_on_operational_failure
    def get_media_attribute_names(self, entity_uuid: str) -> list[str]:
        """Which of an entity's attributes hold media files.

        Only the pointer header is inspected, so this stays cheap for an entity
        that also holds large payloads.
        """
        with self.sql_session as session:
            rows = (session.query(AttributeTable.name, AttributeTable.value_blob)
                    .filter(AttributeTable.entity_uuid == entity_uuid,
                            AttributeTable.data_type == 'blob').all())

        return sorted(row.name for row in rows
                      if blob_store.media_info(row.value_blob) is not None)

    @_retry_on_operational_failure
    def get_entity_attributes(self, _entity: Entity, names: list[str]) -> tuple[Any, ...]:

        entity_uuid = _entity.uuid

        with self.sql_session as session:

            query = session.query(AttributeTable).filter(AttributeTable.entity_uuid == entity_uuid)
            conditions = []
            for n in names:
                conditions.append(AttributeTable.name == n)

            query = query.filter(sqlalchemy.or_(*conditions))

            rows = {row.name: row for row in query.all()}

        # Read values in order of names
        values = []
        for n in names:
            if n not in rows:
                raise AttributeError(f'Attribute "{n}" not found for {_entity}.')
            values.append(_read_attribute_data(_entity, rows[n]))

        return tuple(values)

    @_retry_on_operational_failure
    def get_entity_by_uuid(self,_entarchy: Entarchy, entity_uuid: str) -> Entity:

        with self.sql_session as session:

            query = session.query(EntityTable).filter(EntityTable.uuid == entity_uuid)

            count = query.count()
            if count == 0:
                raise RuntimeError(f'Entity with UUID {entity_uuid} does not exist.')

            entity_row = query.one()
            entity_uuid = entity_row.uuid
            entity_id = entity_row.id
            entity_type = _entarchy.get_entity_type(entity_row.entity_type.name)

        return entity_type(_entarchy, _uuid=entity_uuid, _id=entity_id)

    @_retry_on_operational_failure
    def get_entity_modified_time(self, _entity: Entity) -> datetime.datetime:

        entity_uuid = _entity.uuid

        with self.sql_session as session:

            query = session.query(EntityTable.modified).filter(EntityTable.uuid == entity_uuid)

            count = query.count()
            if count == 0:
                raise KeyError(f'Entity with UUID {entity_uuid} not found in database.')

            row = query.one()

        return row.modified

    @_retry_on_operational_failure
    def get_entities_of_type(self, entity_type: str) -> list[tuple[str, str]]:

        with self.sql_session as session:
            query = (session.query(EntityTable.uuid, EntityTable.id)
                     .order_by(getattr(EntityTable.uuid, 'asc')()))

            # Filter by entity type if provided
            if entity_type is not None:
                query = (query.join(EntityTypeTable)
                         .filter(EntityTypeTable.name == entity_type))

            return [(row.uuid, row.id) for row in query.all()]

    @_retry_on_operational_failure
    def get_entity_parent(self, _entity: Entity) -> Union[tuple[str, str, str], None]:
        entity_uuid = _entity.uuid

        with self.sql_session as session:

            query = session.query(EntityTable).filter(EntityTable.uuid == entity_uuid)

            count = query.count()
            if count == 0:
                return None

            row = query.one()

            if row.parent is None:
                return None
            else:
                return row.parent.entity_type.name, row.parent.uuid, row.parent.id

    # @retry_on_operational_failure
    # def get_link(self, linker: Entity, linked: Entity) -> Union[tuple[str, str, str], None]:
    #
    #     with self.sql_session as session:
    #
    #         query = (session.query(Link)
    #                  .filter(Link.linker_uuid == linker.uuid,
    #                          Link.linked_uuid == linked.uuid))
    #
    #         count = query.count()
    #         if count == 0:
    #             # Create link and entity
    #             new_entity_uuid = str(uuid.uuid4())
    #
    #             # Ensure an entity type for links exists (named 'link')
    #             link_type = session.query(EntityTypeTable).filter(EntityTypeTable.name == 'link').one_or_none()
    #             if link_type is None:
    #                 link_type = EntityTypeTable(name='link')
    #                 session.add(link_type)
    #                 session.flush()  # ensure PK is assigned
    #
    #             # Create the entity row that represents the link (id is a short, readable token)
    #             link_entity_id = f'link-{new_entity_uuid[:8]}'
    #             new_entity_row = EntityTable(
    #                 uuid=new_entity_uuid,
    #                 id=link_entity_id,
    #                 parent_uuid=None,
    #                 entity_type=link_type
    #             )
    #             session.add(new_entity_row)
    #
    #             # Create the Link row that ties the two existing entities and references the new entity
    #             new_link_row = Link(
    #                 linker_uuid=linker_uuid,
    #                 linked_uuid=linked_uuid,
    #                 entity_uuid=new_entity_uuid
    #             )
    #             session.add(new_link_row)
    #
    #             # Commit and reload the created link row
    #             session.commit()
    #             row = session.query(Link).filter(Link.linker_uuid == linker_uuid,
    #                                              Link.linked_uuid == linked_uuid).one()
    #
    #         row = query.one()
    #
    #         if row.entity is None:
    #             return None
    #         else:
    #             return row.entity.entity_type.name, row.entity.uuid, row.entity.id

    @_retry_on_operational_failure
    def has_entity_attribute(self, _entity: Entity, name: str) -> bool:
        entity_uuid = _entity.uuid

        with self.sql_session as session:

            query = session.query(AttributeTable).filter(AttributeTable.entity_uuid == entity_uuid,
                                                         AttributeTable.name == name)

            return query.count() > 0

    @_retry_on_operational_failure
    def set_entity_attribute(self, _entity: Entity, key: str, value: Any):

        self.set_entity_attributes(_entity, [key], [value])

        return True, ''

    @_retry_on_operational_failure
    def set_entity_attributes(self, _entity: Entity, names: list[str], values: list[Any]) -> tuple[bool, str]:

        _entarchy = _entity.entarchy
        entity_uuid = _entity.uuid

        _analysis_uuid = None
        if _entarchy.current_analysis is not None:
            _analysis_uuid = _entarchy.current_analysis.uuid

        with self._write_session() as session:

            attribute_query = session.query(AttributeTable).filter(AttributeTable.entity_uuid == entity_uuid)
            conditions = []
            for n in names:
                conditions.append(AttributeTable.name == n)

            attribute_query = attribute_query.filter(sqlalchemy.or_(*conditions))

            existing_rows = {row.name: row for row in attribute_query.all()}

            # Update existing attributes
            for n, row in existing_rows.items():

                # id and uuid mirror the entity's identity and may never be rewritten
                if n in ('id', 'uuid'):
                    raise RuntimeError(f'Attribute "{n}" is the entity identity '
                                       f'and cannot be modified on {_entity}.')

                # Updates to existing rows are only allowed if attribute is mutable or if digest mode is enabled
                if not (row.mutable or _entity.entarchy.is_in_digest_mode):
                    raise RuntimeError(f'Attribute "{n}" is immutable and cannot be modified on {_entity}.')

                v = values[names.index(n)]

                # Write new value
                _write_attribute_data(_entity, row, v)

            # Add new attributes
            for n in list(set(names) - set(list(existing_rows.keys()))):
                v = values[names.index(n)]

                # Create new attribute row
                new_row = AttributeTable(entity_uuid=entity_uuid,
                                         name=n,
                                         analysis_uuid=_analysis_uuid,
                                         mutable=not _entity.entarchy.is_in_digest_mode)
                session.add(new_row)

                # Write data to row
                _write_attribute_data(_entity, new_row, v)

            # Update entity modified time if triggers are not enabled
            if not self.db_triggers_enabled:
                entity_query = session.query(EntityTable).filter(EntityTable.uuid == entity_uuid)
                entity_row = entity_query.one()  # Check that entity exists
                entity_row.modified = datetime.datetime.now()

            # Commit changes
            self._commit(session)

            return True, ''

    # Link type registry

    def _entity_type_pks(self, session) -> dict[str, int]:
        return {row.name: row.pk for row in session.query(EntityTypeTable).all()}

    def _link_type_from_row(self, row: LinkTypeTable, type_names: dict[int, str]) -> LinkTypeSpec:
        return LinkTypeSpec(
            name=row.name,
            linker=Endpoint(entity_type=type_names.get(row.linker_type_pk),
                            link_type=row.linker_link_type),
            linked=Endpoint(entity_type=type_names.get(row.linked_type_pk),
                            link_type=row.linked_link_type),
            symmetric=bool(row.symmetric),
            cardinality=row.cardinality,
            description=row.description,
        )

    @_retry_on_operational_failure
    def get_link_types(self) -> list[LinkTypeSpec]:
        """Every registered link kind."""
        with self.sql_session as session:
            type_names = {pk: name for name, pk in self._entity_type_pks(session).items()}
            return [self._link_type_from_row(row, type_names)
                    for row in session.query(LinkTypeTable).order_by(LinkTypeTable.name).all()]

    @_retry_on_operational_failure
    def get_link_type(self, name: str) -> Union[LinkTypeSpec, None]:
        with self.sql_session as session:
            row = session.get(LinkTypeTable, name)
            if row is None:
                return None

            type_names = {pk: name for name, pk in self._entity_type_pks(session).items()}
            return self._link_type_from_row(row, type_names)

    @_retry_on_operational_failure
    def add_link_type(self, spec: LinkTypeSpec) -> LinkTypeSpec:
        """Record a new link kind. Fails if the name is already taken."""
        with self._write_session() as session:

            if session.get(LinkTypeTable, spec.name) is not None:
                raise LinkTypeError(f'Link type "{spec.name}" is already defined. '
                                    f'Use redefine_link_type to change it.')

            entity_type_pks = self._entity_type_pks(session)

            def _endpoint_columns(endpoint: Endpoint, label: str) -> tuple:
                if endpoint.link_type is not None:
                    if session.get(LinkTypeTable, endpoint.link_type) is None:
                        raise LinkTypeError(
                            f'The {label} endpoint of "{spec.name}" is declared as a '
                            f'"{endpoint.link_type}" link, but no such link type is '
                            f'defined.')
                    return None, endpoint.link_type

                if endpoint.entity_type is not None:
                    if endpoint.entity_type not in entity_type_pks:
                        raise LinkTypeError(
                            f'The {label} endpoint of "{spec.name}" is declared as '
                            f'"{endpoint.entity_type}", which is not an entity type of '
                            f'this entarchy.')
                    return entity_type_pks[endpoint.entity_type], None

                return None, None

            linker_pk, linker_link = _endpoint_columns(spec.linker, 'linker')
            linked_pk, linked_link = _endpoint_columns(spec.linked, 'linked')

            session.add(LinkTypeTable(
                name=spec.name,
                linker_type_pk=linker_pk, linker_link_type=linker_link,
                linked_type_pk=linked_pk, linked_link_type=linked_link,
                symmetric=spec.symmetric,
                cardinality=spec.cardinality,
                description=spec.description,
            ))
            self._commit(session)

        return spec

    @_retry_on_operational_failure
    def count_links_of_type(self, name: str) -> int:
        with self.sql_session as session:
            return session.query(Link).filter(Link.link_type == name).count()

    # Links

    @_retry_on_operational_failure
    def get_entity_kinds(self, uuids: list[str]) -> dict[str, Endpoint]:
        """What each uuid is, as far as a link constraint is concerned.

        An ordinary entity is its entity type; a link is its kind, because every
        link carries the same entity type and so an entity type says nothing
        useful about one.
        """
        kinds: dict[str, Endpoint] = {}
        unique = list(dict.fromkeys(uuids))

        with self.sql_session as session:
            for start in range(0, len(unique), MAX_BOUND_PARAMETERS):
                batch = unique[start:start + MAX_BOUND_PARAMETERS]

                rows = (session.query(EntityTable.uuid, EntityTypeTable.name)
                        .join(EntityTypeTable)
                        .filter(EntityTable.uuid.in_(batch)).all())
                for entity_uuid, type_name in rows:
                    kinds[entity_uuid] = Endpoint(entity_type=type_name)

                for link_uuid, link_type in (session.query(Link.link_uuid, Link.link_type)
                                             .filter(Link.link_uuid.in_(batch)).all()):
                    kinds[link_uuid] = Endpoint(link_type=link_type)

        return kinds

    @_retry_on_operational_failure
    def add_entity_records(self, records: list[dict]) -> int:
        """Insert entity rows straight from dicts, bypassing the ORM.

        `add_entities` goes through the unit of work, which costs roughly 3 ms an
        entity in ORM overhead once the per-entity commit is gone. Bulk link
        creation writes far too many for that, and its rows are plain data rather
        than Entity objects, so it inserts through the core instead.

        Each record needs uuid, id, parent_uuid and entity_type_name.
        """
        if len(records) == 0:
            return 0

        with self._write_session() as session:
            type_pks = self._entity_type_pks(session)

            rows = []
            for record in records:
                type_name = record['entity_type_name']
                if type_name not in type_pks:
                    raise ValueError(f'Unknown entity type "{type_name}".')

                rows.append({'uuid': record['uuid'], 'id': record['id'],
                             'parent_uuid': record['parent_uuid'],
                             'entity_type_pk': type_pks[type_name],
                             'created': datetime.datetime.now(),
                             'modified': datetime.datetime.now()})

            for chunk in _chunk_by_bound_parameters(rows, len(EntityTable.__table__.columns)):
                session.execute(sqlalchemy.insert(EntityTable), chunk)

            self._commit(session)

        return len(rows)

    @_retry_on_operational_failure
    def add_links(self, records: list[dict]) -> int:
        """Insert link rows. The carrier entities must already exist."""
        if len(records) == 0:
            return 0

        with self._write_session() as session:
            for chunk in _chunk_by_bound_parameters(records, len(Link.__table__.columns)):
                session.execute(sqlalchemy.insert(Link), chunk)
            self._commit(session)

        return len(records)

    @_retry_on_operational_failure
    def get_link_row(self, link_uuid: str) -> Union[dict, None]:
        with self.sql_session as session:
            row = session.get(Link, link_uuid)
            if row is None:
                return None

            return {'link_uuid': row.link_uuid, 'link_type': row.link_type,
                    'linker_uuid': row.linker_uuid, 'linked_uuid': row.linked_uuid}

    @_retry_on_operational_failure
    def find_link(self, link_type: str, linker_uuid: str, linked_uuid: str) -> Union[str, None]:
        """The uuid of an existing link, or None."""
        with self.sql_session as session:
            row = (session.query(Link.link_uuid)
                   .filter(Link.link_type == link_type,
                           Link.linker_uuid == linker_uuid,
                           Link.linked_uuid == linked_uuid).one_or_none())

            return row[0] if row is not None else None

    @_retry_on_operational_failure
    def find_existing_pairs(self, link_type: str, pairs: list[tuple]) -> set:
        """Which of these (linker, linked) pairs already carry this kind."""
        if len(pairs) == 0:
            return set()

        linkers = list({linker for linker, _ in pairs})
        found = set()

        with self.sql_session as session:
            wanted = set(pairs)
            for start in range(0, len(linkers), MAX_BOUND_PARAMETERS):
                batch = linkers[start:start + MAX_BOUND_PARAMETERS]
                for linker, linked in (session.query(Link.linker_uuid, Link.linked_uuid)
                                       .filter(Link.link_type == link_type,
                                               Link.linker_uuid.in_(batch)).all()):
                    if (linker, linked) in wanted:
                        found.add((linker, linked))

        return found

    @_retry_on_operational_failure
    def get_links_for_entity(self, entity_uuid: str, link_type: str = None,
                             direction: str = 'both') -> list[dict]:
        """Every link touching an entity, in either or one direction."""
        with self.sql_session as session:
            query = session.query(Link)

            if direction == 'out':
                query = query.filter(Link.linker_uuid == entity_uuid)
            elif direction == 'in':
                query = query.filter(Link.linked_uuid == entity_uuid)
            else:
                query = query.filter(sqlalchemy.or_(Link.linker_uuid == entity_uuid,
                                                    Link.linked_uuid == entity_uuid))

            if link_type is not None:
                query = query.filter(Link.link_type == link_type)

            return [{'link_uuid': row.link_uuid, 'link_type': row.link_type,
                     'linker_uuid': row.linker_uuid, 'linked_uuid': row.linked_uuid}
                    for row in query.order_by(Link.link_uuid).all()]

    @_retry_on_operational_failure
    def count_links_by_type(self, entity_uuid: str) -> dict[str, int]:
        """How many links of each kind touch an entity, ordered by kind.

        Counted in the database rather than by loading the rows, because the
        callers are a repr and link_types(), and an entity can sit on tens of
        thousands of links - a pairwise correlation over a layer of ROIs gives
        every one of them a link to every other.
        """
        with self.sql_session as session:
            rows = (session.query(Link.link_type, sqlalchemy.func.count())
                    .filter(sqlalchemy.or_(Link.linker_uuid == entity_uuid,
                                           Link.linked_uuid == entity_uuid))
                    .group_by(Link.link_type).order_by(Link.link_type).all())

        return {link_type: int(count) for link_type, count in rows}

    @_retry_on_operational_failure
    def get_links_of_type(self, link_type: str) -> list[dict]:
        with self.sql_session as session:
            return [{'link_uuid': row.link_uuid, 'link_type': row.link_type,
                     'linker_uuid': row.linker_uuid, 'linked_uuid': row.linked_uuid}
                    for row in session.query(Link).filter(
                        Link.link_type == link_type).order_by(Link.link_uuid).all()]

    @_retry_on_operational_failure
    def count_links_per_linker(self, link_type: str, linker_uuids: list[str]) -> dict[str, int]:
        """How many links of a kind each linker already has."""
        counts: dict[str, int] = {}
        unique = list(dict.fromkeys(linker_uuids))

        with self.sql_session as session:
            for start in range(0, len(unique), MAX_BOUND_PARAMETERS):
                batch = unique[start:start + MAX_BOUND_PARAMETERS]
                rows = (session.query(Link.linker_uuid, sqlalchemy.func.count())
                        .filter(Link.link_type == link_type, Link.linker_uuid.in_(batch))
                        .group_by(Link.linker_uuid).all())
                for linker_uuid, count in rows:
                    counts[linker_uuid] = int(count)

        return counts

    @_retry_on_operational_failure
    def remove_links_of_type(self, name: str) -> int:
        """Delete every link of a kind, and the entities carrying them."""
        with self._write_session() as session:
            uuids = [row.link_uuid for row in
                     session.query(Link.link_uuid).filter(Link.link_type == name).all()]

            for start in range(0, len(uuids), MAX_BOUND_PARAMETERS):
                batch = uuids[start:start + MAX_BOUND_PARAMETERS]
                session.query(AttributeTable).filter(
                    AttributeTable.entity_uuid.in_(batch)).delete(synchronize_session=False)
                session.query(Link).filter(
                    Link.link_uuid.in_(batch)).delete(synchronize_session=False)
                session.query(EntityTable).filter(
                    EntityTable.uuid.in_(batch)).delete(synchronize_session=False)

            self._commit(session)

        return len(uuids)

    @_retry_on_operational_failure
    def remove_link_type(self, name: str) -> None:
        """Drop a link kind. The caller is responsible for its links being gone."""
        with self._write_session() as session:
            row = session.get(LinkTypeTable, name)
            if row is not None:
                session.delete(row)
            self._commit(session)

    # Collection related methods

    @_retry_on_operational_failure
    def get_collection_count(self, _collection: Collection, creation_time: datetime.datetime = None) -> int:

        # Fetch result
        with self.sql_session as session:
            query = _build_query_from_collection(_collection, session)

            res = query.count()

            return res

    @_retry_on_operational_failure
    def get_collection_entity_by_index(self, _collection: Collection, index: int, creation_time: datetime.datetime = None) -> tuple[str, str]:

        # Fetch result
        with self.sql_session as session:
            query = _build_query_from_collection(_collection, session)

            res = query.order_by(EntityTable.uuid).offset(index).limit(1).one()

        return res.uuid, res.id

    @_retry_on_operational_failure
    def get_collection_entities_by_slice(self, _collection: Collection, _slice: slice) -> list[tuple[str, str]]:

        # Calculate indices
        count = self.get_collection_count(_collection)
        start, stop, step = _slice.indices(count)

        # Determine the contiguous row range to fetch. For negative steps,
        #  slice.indices returns start >= stop and the range must be flipped
        #  (a negative LIMIT would silently mean "no limit" on SQLite).
        if step > 0:
            offset, limit = start, stop - start
        else:
            offset, limit = stop + 1, start - stop

        if limit <= 0:
            return []

        # Fetch result
        with self.sql_session as session:
            query = _build_query_from_collection(_collection, session)

            res = query.order_by(EntityTable.uuid).offset(offset).limit(limit).all()

        if step < 0:
            res = res[::-1]

        # TODO: there should be a way to directly query the n-th row using 'ROW_NUMBER() % n'
        #        but it's not clear how is would work in SQLAlchemy ORM; figure out later
        return [(r.uuid, r.id) for r in res[::abs(step)]]

    @_retry_on_operational_failure
    def get_collection_parent_uuids(self, _collection: Collection) -> list[tuple[str, str]]:

        # Fetch result
        with self.sql_session as session:

            # Get entity query for collection
            entity_query = _build_query_from_collection(_collection, session)

            # Ordered so that the unordered case is specified rather than left
            #  to the plan. Callers must still pair by uuid - this agrees with
            #  the attribute pivot's order, but nothing should depend on that.
            res = entity_query.order_by(EntityTable.uuid).all()

        return [(r.uuid, r.parent_uuid) for r in res]

    @_retry_on_operational_failure
    def get_collection_attribute_names(self, _collection: Collection) -> list[str]:

        # Fetch result
        with self.sql_session as session:

            # Get entity query for collection
            entity_query = _build_query_from_collection(_collection, session)

            # Get attribute types for requested names
            attribute_query = (session.query(AttributeTable.name)
                               .join(EntityTable)
                               .filter(EntityTable.uuid.in_(_uuids_of(entity_query)))
                               .distinct())

            names = [row.name for row in attribute_query.all()]

        return names

    @_retry_on_operational_failure
    def get_attribute_data_types(self, names: list[str]) -> dict[str, set[str]]:
        """How each named attribute is stored, anywhere in the entarchy.

        A set per name, because one name may be stored with several types -
        `int` for the entities that got a whole number and `float` for the
        rest. Asked of the whole table rather than of a collection, which is
        what makes it an index read; the caller decides whether the ambiguity
        matters.

        Names not stored anywhere are absent from the result rather than
        mapping to an empty set, so a caller can tell "no such attribute" from
        "stored, and here is how".
        """
        types: dict[str, set[str]] = {}
        if len(names) == 0:
            return types

        with self.sql_session as session:
            for row in (session.query(AttributeTable.name, AttributeTable.data_type)
                        .filter(AttributeTable.name.in_(names)).distinct().all()):
                types.setdefault(row.name, set()).add(row.data_type)

        return types

    @_retry_on_operational_failure
    def get_entity_attribute_metadata(self, entity_uuid: str) -> list[tuple[str, str, int]]:
        """(name, data_type, data_size) for every attribute of one entity.

        The shape of a value without reading it. Both columns are written on
        every attribute row, so this answers "what is here and what does it
        cost" from the primary key alone - which is what lets a description of
        an entity holding hundreds of megabytes cost one query.

        `data_size` is bytes as stored: the encoded, compressed container,
        rather than the size in memory once decoded.
        """
        with self.sql_session as session:
            rows = (session.query(AttributeTable.name, AttributeTable.data_type,
                                  AttributeTable.data_size)
                    .filter(AttributeTable.entity_uuid == entity_uuid)
                    .order_by(AttributeTable.name).all())

        return [(row.name, row.data_type, int(row.data_size or 0)) for row in rows]

    @_retry_on_operational_failure
    def get_collection_attribute_metadata(
            self, _collection: Collection) -> list[tuple[str, str, int, int]]:
        """(name, data_type, entity_count, total_size) across a collection.

        Grouped by name and type rather than by name alone, because one name may
        be stored with several types - `int` where a whole number was written and
        `float` elsewhere - and collapsing that here would hide exactly the thing
        worth seeing.
        """
        with self.sql_session as session:
            entity_query = _build_query_from_collection(_collection, session)

            rows = (session.query(AttributeTable.name, AttributeTable.data_type,
                                  sqlalchemy.func.count(),
                                  sqlalchemy.func.sum(AttributeTable.data_size))
                    .filter(AttributeTable.entity_uuid.in_(_uuids_of(entity_query)))
                    .group_by(AttributeTable.name, AttributeTable.data_type)
                    .order_by(AttributeTable.name).all())

        return [(name, data_type, int(count), int(total or 0))
                for name, data_type, count, total in rows]

    @_retry_on_operational_failure
    def get_link_attribute_names(self, entity_uuid: str) -> dict[str, list[str]]:
        """Which attribute names the links touching an entity carry, per kind.

        A kind's names are its shape - `phase_frames` carries `start_index` and
        `end_index` - and that is what a reader meeting an unfamiliar kind wants,
        without reading the links themselves.

        Rides ix_link_linker, ix_link_linked and the attributes primary key, so
        it scales with this entity's links rather than the entarchy's: measured
        1.7 ms at one link, 7.8 ms at 327, 148 ms at 20 000. A LIMIT would not
        help - DISTINCT has to scan before it can stop - so a caller that needs
        a guard should decide from the link count instead.
        """
        with self.sql_session as session:
            rows = (session.query(Link.link_type, AttributeTable.name)
                    .join(AttributeTable, AttributeTable.entity_uuid == Link.link_uuid)
                    .filter(sqlalchemy.or_(Link.linker_uuid == entity_uuid,
                                           Link.linked_uuid == entity_uuid))
                    .distinct().all())

        carried: dict[str, set] = {}
        for link_type, name in rows:
            carried.setdefault(link_type, set()).add(name)

        return {link_type: sorted(names) for link_type, names in sorted(carried.items())}

    @_retry_on_operational_failure
    def count_collection_links_by_type(self, _collection: Collection) -> dict[str, int]:
        """How many links of each kind touch any entity of a collection.

        A link with both ends inside the collection is counted once, which is
        what makes this a count of links rather than of endpoints.
        """
        with self.sql_session as session:
            entity_query = _build_query_from_collection(_collection, session)
            uuids = _uuids_of(entity_query)

            rows = (session.query(Link.link_type, sqlalchemy.func.count(
                        sqlalchemy.distinct(Link.link_uuid)))
                    .filter(sqlalchemy.or_(Link.linker_uuid.in_(uuids),
                                           Link.linked_uuid.in_(uuids)))
                    .group_by(Link.link_type).order_by(Link.link_type).all())

        return {link_type: int(count) for link_type, count in rows}

    @_retry_on_operational_failure
    def count_child_entities(self, entity_uuid: str) -> dict[str, int]:
        """How many children an entity has, by entity type name."""
        with self.sql_session as session:
            rows = (session.query(EntityTypeTable.name, sqlalchemy.func.count())
                    .select_from(EntityTable)
                    .join(EntityTypeTable, EntityTable.entity_type_pk == EntityTypeTable.pk)
                    .filter(EntityTable.parent_uuid == entity_uuid)
                    .group_by(EntityTypeTable.name)
                    .order_by(EntityTypeTable.name).all())

        return {name: int(count) for name, count in rows}

    @_retry_on_operational_failure
    def count_collection_child_entities(self, _collection: Collection) -> dict[str, int]:
        """How many children the entities of a collection have, by type name.

        Asked of parent_uuid directly rather than by building a query per child
        type, so it stays one query however many levels the schema declares.
        """
        with self.sql_session as session:
            entity_query = _build_query_from_collection(_collection, session)

            rows = (session.query(EntityTypeTable.name, sqlalchemy.func.count())
                    .select_from(EntityTable)
                    .join(EntityTypeTable, EntityTable.entity_type_pk == EntityTypeTable.pk)
                    .filter(EntityTable.parent_uuid.in_(_uuids_of(entity_query)))
                    .group_by(EntityTypeTable.name)
                    .order_by(EntityTypeTable.name).all())

        return {name: int(count) for name, count in rows}


    @_retry_on_operational_failure
    def get_collection_attribute_distribution(
            self, _collection: Collection,
            data_types: list[str] = None) -> dict[tuple[str, str], dict[str, Any]]:
        """The range and the spread of the scalar attributes of a collection.

        One entry per (name, data_type) carrying `min`, `max` and `distinct`,
        and for floats also `nan`, `plus_inf` and `minus_inf`. Keyed by name
        *and* type because a name written as `int` on some entities and `float`
        on others has a range in each, and merging them here would hide the
        thing worth seeing.

        NaN and infinity are stored as a flag with a null value column, some
        dialects rejecting them outright - so MIN, MAX and DISTINCT pass over
        them, and a range that let it go at that would silently be a range over
        the finite values alone. They are counted here so the caller can say so.

        Args:
            data_types: which stored types to ask about, defaulting to every
                scalar type. That costs one query each and the entity subquery
                is the expensive part of every one of them, so a caller that
                already knows which types the collection uses should say.
        """
        wanted = [t for t in (data_types if data_types is not None
                              else _DISTRIBUTION_TYPES)
                  if t in _DISTRIBUTION_TYPES]

        distribution: dict[tuple[str, str], dict[str, Any]] = {}
        if len(wanted) == 0:
            return distribution

        with self.sql_session as session:
            entity_query = _build_query_from_collection(_collection, session)
            uuids = _uuids_of(entity_query)

            for data_type in wanted:
                column = getattr(AttributeTable, f'value_{data_type}')
                selected = [AttributeTable.name,
                            sqlalchemy.func.min(column),
                            sqlalchemy.func.max(column),
                            sqlalchemy.func.count(sqlalchemy.distinct(column))]

                if data_type == 'float':
                    # The sign of an infinity lives in the otherwise unused
                    #  value_int column, which is what tells the two ends apart
                    selected += [
                        sqlalchemy.func.sum(sqlalchemy.case(
                            (AttributeTable.float_is_nan, 1), else_=0)),
                        sqlalchemy.func.sum(sqlalchemy.case(
                            (sqlalchemy.and_(AttributeTable.float_is_inf,
                                             AttributeTable.value_int > 0), 1), else_=0)),
                        sqlalchemy.func.sum(sqlalchemy.case(
                            (sqlalchemy.and_(AttributeTable.float_is_inf,
                                             AttributeTable.value_int < 0), 1), else_=0)),
                    ]

                rows = (session.query(*selected)
                        .filter(AttributeTable.entity_uuid.in_(uuids),
                                AttributeTable.data_type == data_type)
                        .group_by(AttributeTable.name)
                        .order_by(AttributeTable.name).all())

                for row in rows:
                    entry = {'min': row[1], 'max': row[2],
                             'distinct': int(row[3] or 0),
                             'nan': 0, 'plus_inf': 0, 'minus_inf': 0}
                    if data_type == 'float':
                        entry['nan'] = int(row[4] or 0)
                        entry['plus_inf'] = int(row[5] or 0)
                        entry['minus_inf'] = int(row[6] or 0)

                    distribution[(row[0], data_type)] = entry

        return distribution

    @_retry_on_operational_failure
    def count_entities_by_type(self) -> dict[str, int]:
        """How many entities of each type the whole entarchy holds.

        A type that has been declared but never used is absent rather than
        zero: this counts what is there, and the hierarchy is what says what
        could be.
        """
        with self.sql_session as session:
            rows = (session.query(EntityTypeTable.name, sqlalchemy.func.count())
                    .select_from(EntityTable)
                    .join(EntityTypeTable, EntityTable.entity_type_pk == EntityTypeTable.pk)
                    .group_by(EntityTypeTable.name)
                    .order_by(EntityTypeTable.name).all())

        return {name: int(count) for name, count in rows}

    @_retry_on_operational_failure
    def get_link_type_totals(self) -> dict[str, dict[str, int]]:
        """How many links of each kind there are, and what they cost.

        `{kind: {'links': n, 'bytes': b}}`. An outer join, so a kind whose links
        carry no attributes still reports its count rather than dropping out;
        the link count is over distinct carriers because the join multiplies a
        link by its attributes.

        Kinds that are registered but unused are absent - `get_link_types()` is
        the registry, this is the census.
        """
        with self.sql_session as session:
            rows = (session.query(
                        Link.link_type,
                        sqlalchemy.func.count(sqlalchemy.distinct(Link.link_uuid)),
                        sqlalchemy.func.sum(AttributeTable.data_size))
                    .outerjoin(AttributeTable,
                               AttributeTable.entity_uuid == Link.link_uuid)
                    .group_by(Link.link_type)
                    .order_by(Link.link_type).all())

        return {link_type: {'links': int(count), 'bytes': int(total or 0)}
                for link_type, count, total in rows}

    @_retry_on_operational_failure
    def get_attribute_storage(self) -> list[tuple[str, str, str, int, int]]:
        """(entity_type, name, data_type, entity_count, total_bytes), entarchy-wide.

        Where the bytes actually are. Grouped rather than summed, so the same
        answer serves both "what does a Recording cost" and "which attribute is
        the largest thing in here" without having to ask twice.

        This is the one query in the describe family that scans the whole
        attributes table rather than riding an index, there being no way to
        total what has not been looked at. It groups down to a row per
        attribute name per entity type, so what comes back stays small however
        large the scan.
        """
        with self.sql_session as session:
            rows = (session.query(EntityTypeTable.name, AttributeTable.name,
                                  AttributeTable.data_type,
                                  sqlalchemy.func.count(),
                                  sqlalchemy.func.sum(AttributeTable.data_size))
                    .select_from(AttributeTable)
                    .join(EntityTable, EntityTable.uuid == AttributeTable.entity_uuid)
                    .join(EntityTypeTable,
                          EntityTable.entity_type_pk == EntityTypeTable.pk)
                    .group_by(EntityTypeTable.name, AttributeTable.name,
                              AttributeTable.data_type)
                    .order_by(EntityTypeTable.name, AttributeTable.name).all())

        return [(type_name, name, data_type, int(count), int(total or 0))
                for type_name, name, data_type, count, total in rows]

    @_retry_on_operational_failure
    def get_collection_attributes(self, _collection: Collection, names: list[str]) -> pd.DataFrame:

        entity_type_name = _collection.entity_type.__name__

        # Fetch result
        with self.sql_session as session:

            # Get entity query for collection
            entity_query = _build_query_from_collection(_collection, session)

            # Which value column holds each requested attribute.
            #
            #  Asked of the whole table rather than of the collection, which is
            #  the difference between reading an index and walking every
            #  attribute row of every entity in the collection to test its
            #  membership: 31 ms against 1024 ms on 27 000 ROIs, and it used to
            #  be the largest single cost of a DataFrame read.
            #
            #  It gives the same answer. The collection's rows are a subset of
            #  the table's, so a name stored with one type everywhere has that
            #  type here too. Only a name stored with several types anywhere
            #  needs the narrower question asked, and that is the rare case the
            #  warning below exists for.
            global_types: dict[str, set] = {}
            for row in (session.query(AttributeTable.name, AttributeTable.data_type)
                        .filter(AttributeTable.name.in_(names)).distinct().all()):
                global_types.setdefault(row.name, set()).add(row.data_type)

            unknown_names = [n for n in names if n not in global_types]
            if len(unknown_names) > 0:
                raise AttributeError(f'Attribute(s) {unknown_names} not found '
                                     f'on any entity in {_collection}.')

            attribute_types = {n: next(iter(global_types[n])) for n in names}

            ambiguous_names = [n for n in names if len(global_types[n]) > 1]
            if len(ambiguous_names) > 0:
                # Only now is it worth asking what this collection actually holds
                resolved: dict[str, str] = {}
                for row in (session.query(AttributeTable.name, AttributeTable.data_type)
                            .filter(AttributeTable.name.in_(ambiguous_names))
                            .join(EntityTable)
                            .filter(EntityTable.uuid.in_(_uuids_of(entity_query)))
                            .join(EntityTypeTable)
                            .filter(EntityTypeTable.name == entity_type_name)
                            .distinct().all()):
                    if row.name in resolved and resolved[row.name] != row.data_type:
                        # TODO: add runtime resolution of problem
                        #        option: always use scalars where available
                        warnings.warn(f'Attribute "{row.name}" has multiple data types in the selected collection. '
                                      f'Using {row.data_type} (not {resolved[row.name]}).',
                                      RuntimeWarning)

                    resolved[row.name] = row.data_type

                attribute_types.update(resolved)

            # Construct query to fetch attributes
            #  Build cases which return correct value field based on attr_name's data_type
            cases = []
            column_labels = ['uuid']
            float_attribute_names = []
            for n in names:
                # Get data type
                data_type = attribute_types[n]

                # Use the appropriate column for the data_type
                cases.append(
                    sqlalchemy.func.max(sqlalchemy.case(
                        (AttributeTable.name == n, getattr(AttributeTable, f'value_{data_type}')),
                        else_=None)).label(n)
                )
                column_labels.append(n)

                # For float attributes, also fetch the nan/inf marker columns.
                #  Special float values are stored as NULL + flag (value_int carries
                #  the sign of inf) and would otherwise be lost in collection reads.
                if data_type == 'float':
                    float_attribute_names.append(n)
                    for flag_col, flag_label in ((AttributeTable.float_is_nan, f'{n}__isnan'),
                                                 (AttributeTable.float_is_inf, f'{n}__isinf'),
                                                 (AttributeTable.value_int, f'{n}__infsign')):
                        cases.append(sqlalchemy.func.max(sqlalchemy.case(
                            (AttributeTable.name == n, flag_col), else_=None)).label(flag_label))
                        column_labels.append(flag_label)

            # Construct query.
            #  Restricted to the requested names: a CASE is built for each of
            #  them, but without this the join still walks every other attribute
            #  row of every entity only to have each CASE discard it. Entities
            #  accumulate attributes as analyses add them, so the ratio gets
            #  worse with the age of an entarchy - on 27 000 entities holding 73
            #  attributes, reading 5 of them went from 4.45 s to 1.17 s. Reading
            #  every attribute of a narrow entity costs 10-15% instead, since
            #  the restriction can then eliminate nothing. That is the trade.
            #
            #  The restriction belongs in the join condition and the join has to
            #  be an outer one. Attributes are per entity rather than per type,
            #  so an entity may have none of the requested names; as an inner
            #  join with the names in the WHERE clause, such an entity produces
            #  no rows at all and silently drops out of the result, where before
            #  it appeared with NaN. The outer join costs nothing measurable
            #  over the inner one.
            attribute_query = (
                session.query(
                    EntityTable.uuid,
                    *cases
                )
                .outerjoin(AttributeTable,
                           sqlalchemy.and_(EntityTable.uuid == AttributeTable.entity_uuid,
                                           AttributeTable.name.in_(names)))
                .filter(EntityTable.uuid.in_(_uuids_of(entity_query)))
                .group_by(EntityTable.uuid, EntityTable.id)
                # SQLite emits groups in key order as a side effect of grouping
                #  and MySQL 8 does not, so say it. Collection.sort() reorders
                #  afterwards; this only fixes what "unsorted" means.
                .order_by(EntityTable.uuid)
            )

        # Create DataFrame from query result
        df = pd.DataFrame(columns=column_labels, data=attribute_query.all())

        # A requested name that no entity in this collection has.
        #
        #  Read off the pivot, which has just answered it, rather than asked
        #  beforehand - asking meant testing collection membership for every
        #  attribute row in the collection, and that was the single largest cost
        #  of this function. Every value column is written non-NULL, so an
        #  all-NULL column means nothing matched; floats are the exception,
        #  since NaN and Inf are stored as NULL plus a marker, and the marker
        #  columns tell the two apart.
        absent_names = []
        for n in names:
            if df[n].notna().any():
                continue
            if n in float_attribute_names and (df[f'{n}__isnan'].notna().any()
                                               or df[f'{n}__isinf'].notna().any()):
                continue
            absent_names.append(n)

        if len(absent_names) > 0:
            raise AttributeError(f'Attribute(s) {absent_names} not found '
                                 f'on any entity in {_collection}.')

        # Convert all types correctly (default result will likely contain bytestring values
        for n in names:

            # Get type
            data_type = attribute_types[n]

            # Cast to type
            try:
                # Use pandas extension types.
                #  This avoids issues with None values in integer columns
                if data_type == 'int':
                    df[n] = df[n].astype(pd.Int64Dtype())
                elif data_type == 'float':
                    df[n] = df[n].astype(pd.Float64Dtype())
                elif data_type == 'str':
                    df[n] = df[n].astype(pd.StringDtype())
                elif data_type == 'bool':
                    df[n] = df[n].astype(pd.BooleanDtype())
                elif data_type == 'date':
                    df[n] = pd.to_datetime(df[n].apply(lambda s: s.decode() if isinstance(s, bytes) else s),
                                           format='%Y-%m-%d')
                elif data_type == 'datetime':
                    df[n] = pd.to_datetime(df[n].apply(lambda s: s.decode() if isinstance(s, bytes) else s),
                                           format='%Y-%m-%d %H:%M:%S')
                # Load blobs
                elif data_type == 'blob':
                    df[n] = df[n].apply(lambda s: _deserialize(s, _collection.entarchy) if s is not None else None)

            except ValueError as e:
                raise RuntimeError(f'Failed to cast attribute {n} to type {data_type}') from e

        # Reconstruct special float values (nan/inf) from their marker columns
        for n in float_attribute_names:
            isnan = df.pop(f'{n}__isnan').astype('boolean').fillna(False).to_numpy(dtype=bool)
            isinf = df.pop(f'{n}__isinf').astype('boolean').fillna(False).to_numpy(dtype=bool)
            infsign = df.pop(f'{n}__infsign').to_numpy(dtype='float64', na_value=np.nan)

            if isnan.any() or isinf.any():
                # Fall back to a plain float64 column so nan/inf are representable
                #  (in pd.Float64Dtype both would collapse to <NA>). Entities that
                #  do not have the attribute at all are nan in this representation.
                values = df[n].to_numpy(dtype='float64', na_value=np.nan)
                values[isinf] = np.where(infsign[isinf] < 0, -np.inf, np.inf)
                df[n] = values

        # Set row index to primary key
        df.set_index('uuid', drop=True, inplace=True)

        return df

    def set_collection_attributes(self, _collection: Collection, df: pd.DataFrame) -> None:
        self.set_attributes_by_uuid(_collection.entarchy, df)

    @_retry_on_operational_failure
    def set_attributes_by_uuid(self, ent: Entarchy, df: pd.DataFrame) -> None:
        """Write attributes for entities addressed by uuid.

        The rows are identified by the frame's index, so this needs no
        collection - which is what lets links use the same bulk path, since a
        freshly created set of links is not a query result.
        """

        if df.empty or len(df.columns) == 0:
            return

        _dtypes = ['str', 'float', 'int', 'date', 'datetime', 'bool', 'blob']

        _analysis_uuid = None
        if ent.current_analysis is not None:
            _analysis_uuid = ent.current_analysis.uuid

        # Do an upsert for each attribute to be updated
        for attr_name in df.columns:

            # id and uuid mirror the entity's identity and may never be rewritten
            if attr_name in ('id', 'uuid'):
                raise RuntimeError(f'Attribute "{attr_name}" is the entity identity '
                                   f'and cannot be modified through a collection.')

            # Create df for insert
            df_insert = pd.DataFrame(df[attr_name])

            # Determine data type
            dtype = str(df_insert[attr_name].dtype).lower()
            if 'int' in dtype:
                data_type_str = 'int'
            elif 'float' in dtype:
                data_type_str = 'float'
            elif 'bool' in dtype:
                data_type_str = 'bool'
            elif 'str' in dtype:
                data_type_str = 'str'

            # If column information does not contain a specific data type, try to figure it out anyway
            elif 'object' in dtype:

                # Use first row to determine type
                _dt = type(df_insert[attr_name].head(1).values[0])
                if _dt is str:
                    data_type_str = 'str'
                elif _dt is datetime.date:
                    data_type_str = 'date'
                elif _dt is datetime.datetime:
                    data_type_str = 'datetime'
                else:
                    data_type_str = 'blob'

            # Fallback: make it opaque
            else:
                data_type_str = 'blob'

            # Add PK set
            df_insert['entity_uuid'] = df_insert.index
            df_insert['name'] = attr_name
            df_insert['data_type'] = data_type_str
            df_insert['analysis_uuid'] = _analysis_uuid

            # Handle special float values: nan/inf are stored as NULL plus marker flags
            #  (value_int carries the sign of inf), mirroring _write_attribute_data
            if data_type_str == 'float':
                _values = df_insert[attr_name].to_numpy(dtype='float64', na_value=np.nan)
                _isinf = np.isinf(_values)
                _isnan = np.isnan(_values)
                df_insert['float_is_nan'] = _isnan
                df_insert['float_is_inf'] = _isinf

                # These columns mix None with numbers. Assigning a plain list lets
                #  pandas infer float64 and turn every None into NaN, which SQLite
                #  stores happily but MySQL rejects outright ("nan can not be used
                #  with MySQL"), so the None values are preserved explicitly.
                df_insert['value_int'] = pd.Series(
                    [int(np.sign(v)) if np.isinf(v) else None for v in _values],
                    index=df_insert.index, dtype=object)
                df_insert[attr_name] = pd.Series(
                    [None if (np.isnan(v) or np.isinf(v)) else float(v) for v in _values],
                    index=df_insert.index, dtype=object)
            else:
                # Reset markers in case the attribute previously held special float values
                df_insert['float_is_nan'] = False
                df_insert['float_is_inf'] = False

            # Serialize data
            if data_type_str == 'blob':
                # Encode per row, since where a value is stored depends on how
                #  big it turns out to be
                df_insert[attr_name] = df_insert.apply(
                    lambda series: _store_blob(series[attr_name], ent.path,
                                               ent.max_blob_size,
                                               series['entity_uuid'], attr_name),
                    axis=1)

            # Save attribute size
            def _get_attribute_size(series: pd.Series) -> int:
                if data_type_str == 'int':
                    return 8
                elif data_type_str == 'float':
                    return 8
                elif data_type_str == 'bool':
                    return 1
                elif data_type_str == 'date':
                    return 3
                elif data_type_str == 'datetime':
                    return 8
                elif data_type_str == 'str':
                    return len(series[attr_name].encode('utf-8'))
                elif data_type_str == 'blob':
                    # Already encoded above; a pointer's size is the file it names
                    return _stored_size(series[attr_name], ent.path)
                else:
                    raise RuntimeError(f'Unsupported data type {data_type_str}.')

            df_insert['data_size'] = df_insert.apply(_get_attribute_size, axis=1)

            # Rename value column
            df_insert.rename(columns={attr_name: f'value_{data_type_str}'}, inplace=True)

            if not self.db_triggers_enabled:
                df_insert['modified'] = datetime.datetime.now()

            # Perform upsert, in chunks small enough for the dialect to accept
            with self._write_session() as session:
                insert_attr_data = df_insert.to_dict('records')

                # Sized from the table, not the DataFrame: SQLAlchemy fills in
                #  columns the DataFrame never carries (mutable, created) from the
                #  model defaults, and those bind parameters too
                for chunk in _chunk_by_bound_parameters(insert_attr_data,
                                                        len(AttributeTable.__table__.columns)):
                    insert_stmt = self._insert_statement(chunk)
                    proposed = self._inserted_values(insert_stmt)

                    update_attr_data = {
                        f'value_{data_type_str}': getattr(proposed, f'value_{data_type_str}'),
                        # On update, reset all other value fields to None:
                        **{f'value_{dt}': None for dt in list(set(_dtypes) - {data_type_str})},
                        'data_type': data_type_str,
                        'data_size': proposed.data_size,
                        'analysis_uuid': proposed.analysis_uuid,
                        'modified': proposed.modified,
                        'float_is_nan': proposed.float_is_nan,
                        'float_is_inf': proposed.float_is_inf,
                    }

                    # For floats, value_int carries the sign of inf values and must
                    #  override the generic value-column reset above
                    if data_type_str == 'float':
                        update_attr_data['value_int'] = proposed.value_int

                    # Add modified time update if triggers are not enabled
                    if not self.db_triggers_enabled:
                        update_attr_data['modified'] = datetime.datetime.now()

                    # Execute upsert
                    session.execute(self._upsert_statement(insert_stmt, update_attr_data))

                self._commit(session)

        if not self.db_triggers_enabled:
            with self._write_session() as session:
                # IN binds one parameter per uuid, so this needs chunking too
                uuids = list(df.index)
                for start in range(0, len(uuids), MAX_BOUND_PARAMETERS):
                    session.query(EntityTable).filter(
                        EntityTable.uuid.in_(uuids[start:start + MAX_BOUND_PARAMETERS])
                    ).update(
                        {EntityTable.modified: datetime.datetime.now()},
                        synchronize_session=False,
                    )
                self._commit(session)

    def open(self):
        # Just access the property to create the engine if it doesn't exist yet
        # _ = self.sql_engine
        pass

    def close(self):
        # Only tear down what actually exists; close() must be safe to call
        #  regardless of whether a session/engine was ever created
        if getattr(self, '_sql_session', None) is not None:
            self._sql_session.close()
            self._sql_session = None
        if getattr(self, '_sql_engine', None) is not None:
            self._sql_engine.dispose()
            self._sql_engine = None
