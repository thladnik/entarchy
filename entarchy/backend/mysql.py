import hashlib
import io
import math
import operator
import os.path
import pathlib
import pickle
import datetime
from typing import Any, Callable, List, Union

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import Index, ForeignKey, String, create_engine, BigInteger, Double
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.exc import OperationalError

from .backend import Backend
from .. import AnalysisEntity, Collection, Entarchy, Entity


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
    parent_uuid: Mapped[uuid] = mapped_column(ForeignKey('entities.uuid'), nullable=True)
    entity_type_pk: Mapped[int] = mapped_column(ForeignKey('entity_types.pk'))

    id: Mapped[str] = mapped_column(String(500))

    # Many-to-One
    entity_type: Mapped['EntityTypeTable'] = relationship('EntityTypeTable', back_populates='entities')
    parent: Mapped['EntityTable'] = relationship('EntityTable', back_populates='children', remote_side=[uuid])

    # One-to-Many
    children: Mapped[List['EntityTable']] = relationship('EntityTable', back_populates='parent', remote_side=[parent_uuid])
    attributes: Mapped[List['AttributeTable']] = relationship('AttributeTable', back_populates='entity')

    created: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)
    modified: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now, onupdate=datetime.datetime.now)

    __table_args__ = (
        Index('ix_unique_id_per_parent_and_entity_type', 'parent_uuid', 'entity_type_pk', 'id', unique=True),
    )

    def __repr__(self):
        return f"<{self.entity_type.name}Row(id={self.id}, parent={self.parent})>"


class Link(Base):
    __tablename__ = 'links'

    linker_uuid: Mapped[str] = mapped_column(String(36), ForeignKey('entities.uuid'), primary_key=True)
    linker = relationship('EntityTable', foreign_keys=[linker_uuid])
    linked_uuid: Mapped[str] = mapped_column(String(36), ForeignKey('entities.uuid'), primary_key=True)
    linked = relationship('EntityTable', foreign_keys=[linked_uuid])
    entity_uuid: Mapped[str] = mapped_column(String(36), ForeignKey('entities.uuid'), nullable=True)
    entity = relationship('EntityTable', foreign_keys=[entity_uuid])

    created: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)
    modified: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now, onupdate=datetime.datetime.now)


class AttributeTable(Base):
    __tablename__ = 'attributes'

    entity_uuid: Mapped[str] = mapped_column(String(36), ForeignKey('entities.uuid'), primary_key=True)
    entity: Mapped['EntityTable'] = relationship('EntityTable', foreign_keys=[entity_uuid], back_populates='attributes')
    analysis_uuid: Mapped[str] = mapped_column(String(36), nullable=True)

    name: Mapped[str] = mapped_column(String(500), primary_key=True, index=True)

    value_str: Mapped[str] = mapped_column(String(500), nullable=True)
    value_int: Mapped[int] = mapped_column(BigInteger(), nullable=True)
    value_float: Mapped[float] = mapped_column(Double(), nullable=True)
    value_bool: Mapped[bool] = mapped_column(nullable=True)
    value_date: Mapped[datetime.date] = mapped_column(nullable=True)
    value_datetime: Mapped[datetime.datetime] = mapped_column(nullable=True)
    value_blob: Mapped[bytes] = mapped_column(LONGBLOB, nullable=True)
    data_type: Mapped[str] = mapped_column(String(500), nullable=True)

    float_is_nan: Mapped[bool] = mapped_column(default=False)
    float_is_inf: Mapped[bool] = mapped_column(default=False)

    mutable = mapped_column(sqlalchemy.Boolean, default=True)

    created: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now)
    modified: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.now, onupdate=datetime.datetime.now)

    __table_args__ = (
        Index('ix_unique_name_per_entity_uuid', 'entity_uuid', 'name', unique=True),
    )

    def __repr__(self):
        return f"<Attribute({self.name}, {self.entity})>"


def retry_on_operational_failure(fun: Callable, retry_num: int = 3) -> Callable:
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
                err = e
                i += 1

            except Exception as e:
                err = e
                raise e

        if err is not None:
            raise err

    return _wrapper


def _build_query_from_collection(_collection: Collection,
                                 _session: sqlalchemy.orm.Session
                                 ) -> sqlalchemy.orm.Query:

    entity_type_name = _collection.entity_type.__name__
    as_tree = _collection.as_tree
    creation_time = _collection.init_time

    # Create base query
    _query = _session.query(EntityTable).join(EntityTypeTable).filter(EntityTypeTable.name == entity_type_name)
    _query = _query.filter(EntityTable.created <= creation_time)

    # Apply filters generated from the abstract syntax tree
    if len(as_tree) == 0:
        return _query

    filters = _generate_attribute_filters(entity_type_name, _session, as_tree)

    return _query.filter(filters)


def _generate_attribute_filters(entity_type_name: str,
                                _session: sqlalchemy.orm.Session,
                                as_tree: dict[str, ...]) -> Any:

    _operator = as_tree['operator'].upper()

    # Handle connectives
    if _operator in ('AND', 'OR'):
        # TODO: implement XOR? Very costly to do like this in SQL // see entarchy.core.query.combine_trees
        #        Requires at least OR + 2xAND + NOT
        _op_fun = {'AND': sqlalchemy.and_, 'OR': sqlalchemy.or_}[_operator]
        return _op_fun(_generate_attribute_filters(entity_type_name, _session, as_tree['left_operand']),
                       _generate_attribute_filters(entity_type_name, _session, as_tree['right_operand']))

    # Handle comparisons
    elif _operator in ('IN', '<=', '<', '==', '>', '>='):

        name = as_tree['left_operand']
        value = as_tree['right_operand']

        if _operator == 'IN':
            raise NotImplementedError(f'IN operator not implemented for MySQL backend yet')
            # if not isinstance(value, list):
            #     raise ValueError('Operand after IN statement should be a list of values')
            #
            # attribute_value_col = getattr(AttributeTable, f'value_{value[0].__class__.__name__}')
            #
            # comparison = attribute_value_col.in_(value)
        else:
            # Determine the correct column based on value type
            attribute_value_col = getattr(AttributeTable, f'value_{value.__class__.__name__}')

            _op_fun = {
                '<': operator.lt,
                '<=': operator.le,
                '==': operator.eq,
                '>=': operator.ge,
                '>': operator.gt
            }[_operator]

            comparison = _op_fun(attribute_value_col, value)

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
                parent_level = get_entity_type_ancestor_distance(_session, entity_type_name, parent_entity_type_name)

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
        subquery = (_session.query(AttributeTable.entity_uuid)
                    .filter(AttributeTable.name == as_tree['right_operand'])
                    .subquery())

    elif _operator == 'NOT':
        return sqlalchemy.not_(_generate_attribute_filters(entity_type_name, _session, as_tree['right_operand']))

    # Fallback
    else:
        print(f'Unknown unary operator: {_operator}', as_tree)
        raise ValueError('Unexpected operator in the expression tree')

    # Return the `IN` filter to apply to the main query
    return EntityTable.uuid.in_(_session.query(subquery.c.entity_uuid))


def get_entity_type_ancestor_distance(session: sqlalchemy.orm.Session,
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


def _get_namehash(name: str) -> str:
    return hashlib.sha224(name.encode()).hexdigest()


# def _get_attribute_fp(_entity: Entity, row: AttributeTable, _format) -> tuple[str, str]:
#
#     _uuid = _entity.uuid.replace('-', '')
#     _shards = [_uuid[4*i:4*(i+1)] for i in range(8)]  # Create path shards from uuid (8x4 characters for uuid4)
#     fp = os.path.join(_entity.entarchy.path, 'ext', *_shards)
#
#     return pathlib.Path(fp).as_posix(), f'{_get_namehash(row.name)}.{_format}'

def _get_attribute_fp(_entity: Entity, name: str, _format) -> tuple[str, str]:

    _uuid = _entity.uuid.replace('-', '')
    _shards = [_uuid[4*i:4*(i+1)] for i in range(8)]  # Create path shards from uuid (8x4 characters for uuid4)
    fp = os.path.join(_entity.entarchy.path, 'ext', *_shards)

    return pathlib.Path(fp).as_posix(), f'{_get_namehash(name)}.{_format}'


def _read_attribute_data(_entity: Entity, row: AttributeTable):
    if row.data_type is None:
        raise ValueError('Attribute data type is None.')

    # Load blob
    if row.data_type == 'blob':
        ser: Serializer = pickle.loads(row.value_blob)
        return ser.deserialize(_entity.entarchy)

    # if row.data_type.startswith('blob'):
        # data_type, _format = row.data_type.split('::')
        #
        # if data_type == 'blob_ext':
        #
        #     fp, fn = _get_attribute_fp(_entity, row.name, _format)
        #
        #     with open(f'{fp}/{fn}', 'rb') as f:
        #         data_serial = f.read()
        #
        # else:
        #     data_serial = row.value_blob
        #
        # return _deserialize(data_serial, _format)

    # Otherwise load from this row based on data type
    val = getattr(row, f'value_{row.data_type}')

    # Check for inf and NaNs
    if row.data_type == 'float' and val is None:
        if row.float_is_inf:
            return float('inf')
        elif row.float_is_nan:
            return float('nan')

    return val


class Serializer(object):

    _type: type
    _store: str
    _data: bytes

    def __repr__(self):
        return f'Serializer({self._type}, {self._store})'

    def serialize(self, _entity: Entity, name: str, data: Any):

        self._type = type(data)

        if isinstance(data, bytes):
            self._data = data
        # elif isinstance(data, list):
        #     self._data = pickle.dumps(data)
        # elif isinstance(data, dict):
        #     self._data = pickle.dumps(data)
        elif isinstance(data, np.ndarray):
            with io.BytesIO() as buffer:
                np.lib.format.write_array(buffer, data, allow_pickle=True)
                buffer.seek(0)
                self._data = buffer.read()
        else:
            self._data = pickle.dumps(data)

        if self._data.__sizeof__() >= _entity.entarchy.max_blob_size:

            # Save to file
            _format = 'npy' if self._type is np.ndarray else 'pickle'

            fp, fn = _get_attribute_fp(_entity, name, _format)

            os.makedirs(fp, exist_ok=True)

            fullpath = f'{fp}/{fn}'
            self._store = pathlib.Path(fullpath).relative_to(_entity.entarchy.path)

            with open(fullpath, 'wb') as f:
                f.write(self._data)
                del self._data
                # print('Saved large blob to external file:', fullpath)
                # print('Serializer size:', self.__sizeof__())

        else:
            self._store = 'internal'

    def deserialize(self, _entarchy: Entarchy) -> Any:

        # print(f'Deserialize {self}')
        # Read from file if needed
        if self._store != 'internal':
            # Load from file
            # print(f'Load from path {self._store}')
            with open(os.path.join(_entarchy.path, self._store), 'rb') as f:
                self._data = f.read()

        # Return data according to original type
        if self._type is bytes:
            return self._data
        # elif self._type is list:
        #     return pickle.loads(self._data)
        # elif self._type is dict:
        #     return pickle.loads(self._data)
        elif self._type is np.ndarray:
            with io.BytesIO(self._data) as buffer:
                buffer.seek(0)
                return np.lib.format.read_array(buffer, allow_pickle=True)
        else:
            return pickle.loads(self._data)
            raise TypeError(f'Unsupported data type during deserialization: {self._type}')


def _write_attribute_data(_entity: Entity, row: AttributeTable, data: Any):

    # TODO: in future version, information about data type byte number should be included in data_type column
    #  This way the exact data type can be restored upon read (e.g. int8, int16, float32, float64, etc.)
    #  This would mean that python native scalars may be stored as regular 64bit,
    #  while numpy scalars get variable sizes.
    #  This won't affect actual storage though, as the database will use the same column types (bigint, double) anyway.

    # Get corresponding builtin python scalar type for numpy scalars
    if isinstance(data, np.generic):
        data = data.item()

    # If previous data type war float, reset flags
    if row.data_type == 'float':
        row.float_is_inf = False
        row.float_is_nan = False

    # Set (potential) previous value to None
    row.__setattr__(f'value_{row.data_type}', None)

    # TODO: with ext storage for blobs, we should also delete previous files on disk

    # Handle scalars and datetime values
    if type(data) in (str, float, int, bool, datetime.date, datetime.datetime):

        # Set value type
        data_type_map = {str: 'str', float: 'float', int: 'int',
                         bool: 'bool', datetime.date: 'date', datetime.datetime: 'datetime'}
        data_type = data_type_map.get(type(data))

        # Some SQL dialects don't support inf float values
        if data_type == 'float' and math.isinf(data):
            row.float_is_inf = True
            data = None
        elif data_type == 'float' and math.isnan(data):
            row.float_is_nan = True
            data = None
    else:
        data_type = 'blob'

    # Set value on corresponding column based on type
    if data_type == 'blob':

        # Create serializer
        ser = Serializer()
        ser.serialize(_entity, row.name, data)

        # Serialize the serializer
        data = pickle.dumps(ser)

    row.data_type = data_type

    # Set data
    row.__setattr__(f'value_{data_type}', data)


def _deserialize(data: bytes, _entarchy: Entarchy) -> Any:

    ser: Serializer = pickle.loads(data)
    return ser.deserialize(_entarchy)


class MySQLBackend(Backend):
    _sql_engine: sqlalchemy.Engine | None = None
    _sql_session: sqlalchemy.orm.Session | None = None
    _db_triggers_enabled: bool = None

    def __init__(self, dbname: str, dbhost: str, dbuser: str, dbpassword: str = None, debug: bool = False):

        # Get connection parameters
        if dbhost is None:
            dbname = input(f'MySQL host name [default: "localhost"]: ')
            if dbname == '':
                dbname = 'localhost'

        if dbname is None:
            while True:
                dbname = input(f'New database schema name on host "{dbhost}": ')
                if dbname == '':
                    print('Database schema name cannot be empty.')
                else:
                    break

        if dbuser is None:
            dbuser = input(f'User name for database schema "{dbname}" [default: entarchy_user]: ')
            if dbuser == '':
                dbuser = 'entarchy_user'

        if dbpassword is None:
            import getpass
            dbpassword = getpass.getpass(f'Password for user {dbuser}: ')

        self._config = {
            'dbname': dbname,
            'dbhost': dbhost,
            'dbuser': dbuser,
            'dbpassword': dbpassword,
        }

        self.debug = debug

    @property
    def dbhost(self) -> str:
        return self._config['dbhost']

    @property
    def dbname(self) -> str:
        return self._config['dbname']

    @property
    def dbuser(self) -> str:
        return self._config['dbuser']

    @property
    def dbpassword(self) -> str:
        return self._config['dbpassword']

    @property
    def sql_engine(self):
        if not hasattr(self, '_sql_engine') or self._sql_engine is None:
            self._sql_engine = create_engine(f'mysql+pymysql://'
                                             f'{self.dbuser}:{self.dbpassword}'
                                             f'@{self.dbhost}/{self.dbname}',
                                             echo=self.debug,
                                             pool_size=1,
                                             pool_recycle=60,
                                             pool_pre_ping=True,
                                             )

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
        if self._db_triggers_enabled is None:

            # Check if triggers are enabled
            if self._db_triggers_enabled is None:
                with sqlalchemy.Connection(self._sql_engine) as conn:
                    res = conn.execute(
                        sqlalchemy.text(
                            'SELECT TRIGGER_NAME '
                            'FROM information_schema.TRIGGERS '
                            f'WHERE TRIGGER_SCHEMA = \'{self.dbname}\''
                            'AND TRIGGER_NAME = \'attributes_touch_entities_ai\''
                        )
                    )
                    self._db_triggers_enabled = len(res.fetchall()) > 0

        return self._db_triggers_enabled

    @retry_on_operational_failure
    def create(self) -> bool:

        # Create schema
        print(f'> Create database {self.dbname}')
        engine = create_engine(f'mysql+pymysql://{self.dbuser}:{self.dbpassword}@{self.dbhost}', echo=self.debug)
        with engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'CREATE SCHEMA IF NOT EXISTS {self.dbname}'))
        engine.dispose()

        # Create tables
        print('> Create tables')
        Base.metadata.create_all(self.sql_engine)

        print('> Create triggers')
        with self.sql_engine.connect() as connection:
            try:
                # TODO: this needs to use non-utc datetime
                connection.execute(
                    sqlalchemy.text(
                        "CREATE TRIGGER attributes_touch_entities_ai\n"
                        "AFTER INSERT ON attributes FOR EACH ROW\n"
                        "UPDATE entities\n"
                        "SET modified = UTC_TIMESTAMP(6)\n"
                        "WHERE entities.uuid = NEW.entity_uuid\n"
                    )
                )
                connection.execute(
                    sqlalchemy.text(
                        "CREATE TRIGGER attributes_touch_entities_au\n"
                        "AFTER UPDATE ON attributes FOR EACH ROW\n"
                        "UPDATE entities\n"
                        "SET modified = UTC_TIMESTAMP(6)\n"
                        "WHERE entities.uuid = NEW.entity_uuid\n"
                    )
                )
                connection.execute(
                    sqlalchemy.text(
                        "CREATE TRIGGER attributes_touch_entities_ad\n"
                        "AFTER DELETE ON attributes FOR EACH ROW\n"
                        "UPDATE entities\n"
                        "SET modified = UTC_TIMESTAMP(6)\n"
                        "WHERE entities.uuid = OLD.entity_uuid\n"
                    )
                )
            except:
                # If this fails after successful creation of tables, the most likely reason is detailed here:
                #  https://stackoverflow.com/a/56390000
                print('WARNING: Failed to create database triggers. '
                      'This may impact performance of attribute updates.')
                print(10 * ' ' + 'A likely cause for this are server security settings '
                                 'or insufficient privileges of the database user.')
                print(10 * ' ' + f'For better performance give global \'SUPER\' privilege to dbuser \'{self.dbuser}\' ')
                print(10 * ' ' + f'OR set log-bin-trust-function-creators=1 in MySQL config and restart the server.')

        return True

    @retry_on_operational_failure
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

    @retry_on_operational_failure
    def delete(self, confirm: bool = False):

        if not confirm:
            raise RuntimeError('Failed to delete backend. Confirmation not provided.')
            return

        print('> Drop schema')
        engine = create_engine(f'mysql+pymysql://{self.dbuser}:{self.dbpassword}@{self.dbhost}')
        with engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'DROP SCHEMA IF EXISTS {self.dbname}'))
            connection.commit()

    # Entity related methods

    @retry_on_operational_failure
    def add_entities(self, _entities: list[Entity]) -> bool:

        with self.sql_session as session:

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
            session.commit()

        return True

    def get_entity_attribute(self, _entity: Entity, name: str) -> Any:

        return self.get_entity_attributes(_entity, [name])[0]

    @retry_on_operational_failure
    def get_entity_attribute_names(self, _entity: Entity) -> list[str]:

        entity_uuid = _entity.uuid

        with self.sql_session as session:

            query = session.query(AttributeTable.name).filter(AttributeTable.entity_uuid == entity_uuid)

            names = [row.name for row in query.all()]

        return names

    @retry_on_operational_failure
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

    @retry_on_operational_failure
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

    @retry_on_operational_failure
    def get_entity_modified_time(self, _entity: Entity) -> datetime.datetime:

        entity_uuid = _entity.uuid

        with self.sql_session as session:

            query = session.query(EntityTable.modified).filter(EntityTable.uuid == entity_uuid)

            count = query.count()
            if count == 0:
                raise KeyError(f'Entity with UUID {entity_uuid} not found in database.')

            row = query.one()

        return row.modified

    @retry_on_operational_failure
    def get_entities_of_type(self, entity_type: str) -> list[tuple[str, str]]:

        with self.sql_session as session:
            query = (session.query(EntityTable.uuid, EntityTable.id)
                     .order_by(getattr(EntityTable.uuid, 'asc')()))

            # Filter by entity type if provided
            if entity_type is not None:
                query = (query.join(EntityTypeTable)
                         .filter(EntityTypeTable.name == entity_type))

            return [(row.uuid, row.id) for row in query.all()]

    @retry_on_operational_failure
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

    @retry_on_operational_failure
    def get_link(self, linker: Entity, linked: Entity) -> Union[tuple[str, str, str], None]:

        with self.sql_session as session:

            query = (session.query(Link)
                     .filter(Link.linker_uuid == linker.uuid,
                             Link.linked_uuid == linked.uuid))

            count = query.count()
            if count == 0:
                # Create link and entity
                new_entity_uuid = str(uuid.uuid4())

                # Ensure an entity type for links exists (named 'link')
                link_type = session.query(EntityTypeTable).filter(EntityTypeTable.name == 'link').one_or_none()
                if link_type is None:
                    link_type = EntityTypeTable(name='link')
                    session.add(link_type)
                    session.flush()  # ensure PK is assigned

                # Create the entity row that represents the link (id is a short, readable token)
                link_entity_id = f'link-{new_entity_uuid[:8]}'
                new_entity_row = EntityTable(
                    uuid=new_entity_uuid,
                    id=link_entity_id,
                    parent_uuid=None,
                    entity_type=link_type
                )
                session.add(new_entity_row)

                # Create the Link row that ties the two existing entities and references the new entity
                new_link_row = Link(
                    linker_uuid=linker_uuid,
                    linked_uuid=linked_uuid,
                    entity_uuid=new_entity_uuid
                )
                session.add(new_link_row)

                # Commit and reload the created link row
                session.commit()
                row = session.query(Link).filter(Link.linker_uuid == linker_uuid,
                                                 Link.linked_uuid == linked_uuid).one()

            row = query.one()

            if row.entity is None:
                return None
            else:
                return row.entity.entity_type.name, row.entity.uuid, row.entity.id

    @retry_on_operational_failure
    def has_entity_attribute(self, _entity: Entity, name: str) -> bool:
        entity_uuid = _entity.uuid

        with self.sql_session as session:

            query = session.query(AttributeTable).filter(AttributeTable.entity_uuid == entity_uuid,
                                                         AttributeTable.name == name)

            return query.count() > 0

    @retry_on_operational_failure
    def set_entity_attribute(self, entity_uuid, key: str, value: Any):

        self.set_entity_attributes(entity_uuid, [key], [value])

        return True, ''

    @retry_on_operational_failure
    def set_entity_attributes(self, _entity: Entity, names: list[str], values: list[Any]) -> tuple[bool, str]:

        _entarchy = _entity.entarchy
        entity_uuid = _entity.uuid

        _analysis_uuid = None
        if _entarchy.current_analysis is not None:
            _analysis_uuid = _entarchy.current_analysis.uuid

        with self.sql_session as session:

            attribute_query = session.query(AttributeTable).filter(AttributeTable.entity_uuid == entity_uuid)
            conditions = []
            for n in names:
                conditions.append(AttributeTable.name == n)

            attribute_query = attribute_query.filter(sqlalchemy.or_(*conditions))

            existing_rows = {row.name: row for row in attribute_query.all()}

            # Update existing attributes
            for n, row in existing_rows.items():

                if not row.mutable:
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
            session.commit()

            return True, ''

    # Collection related methods

    @retry_on_operational_failure
    def get_collection_count(self, _collection: Collection, creation_time: datetime.datetime = None) -> int:

        # Fetch result
        with self.sql_session as session:
            query = _build_query_from_collection(_collection, session)

            res = query.count()

            return res

    @retry_on_operational_failure
    def get_collection_entity_by_index(self, _collection: Collection, index: int, creation_time: datetime.datetime = None) -> tuple[str, str]:

        # Fetch result
        with self.sql_session as session:
            query = _build_query_from_collection(_collection, session)

            res = query.order_by(EntityTable.uuid).offset(index).limit(1).one()

        return res.uuid, res.id

    @retry_on_operational_failure
    def get_collection_entities_by_slice(self, _collection: Collection, _slice: slice) -> list[tuple[str, str]]:

        entity_type_name = _collection.entity_type.__name__

        # Calculate indices
        count = self.get_collection_count(_collection, entity_type_name)
        start, stop, step = _slice.indices(count)

        # Fetch result
        with self.sql_session as session:
            query = _build_query_from_collection(_collection, session)

            res = query.order_by(EntityTable.uuid).offset(start).limit(stop - start).all()

        # TODO: there should be a way to directly query the n-th row using 'ROW_NUMBER() % n'
        #        but it's not clear how is would work in SQLAlchemy ORM; figure out later
        return [(r.uuid, r.id) for r in res[::step]]

    @retry_on_operational_failure
    def get_collection_parent_uuids(self, _collection: Collection) -> list[tuple[str, str]]:

        # Fetch result
        with self.sql_session as session:

            # Get entity query for collection
            entity_query = _build_query_from_collection(_collection, session)

            # res = parent_query.all()
            res = entity_query.all()

        return [(r.uuid, r.parent_uuid) for r in res]

    @retry_on_operational_failure
    def get_collection_attribute_names(self, _collection: Collection) -> list[str]:

        # Fetch result
        with self.sql_session as session:

            # Get entity query for collection
            entity_query = _build_query_from_collection(_collection, session)

            # Get attribute types for requested names
            attribute_query = (session.query(AttributeTable.name)
                               .join(EntityTable)
                               .filter(EntityTable.uuid.in_(entity_query.subquery().primary_key))
                               .distinct())

            names = [row.name for row in attribute_query.all()]

        return names

    @retry_on_operational_failure
    def get_collection_attributes(self, _collection: Collection, names: list[str]) -> pd.DataFrame:

        entity_type_name = _collection.entity_type.__name__

        # Fetch result
        with self.sql_session as session:

            # Get entity query for collection
            entity_query = _build_query_from_collection(_collection, session)

            # Get attribute types for requested names
            attribute_types = {}
            distinct_attribute_query = (session.query(AttributeTable.name, AttributeTable.data_type)
                                        .join(EntityTable)
                                        .filter(EntityTable.uuid.in_(entity_query.subquery().primary_key))
                                        .join(EntityTypeTable).filter(EntityTypeTable.name == entity_type_name)
                                        .distinct())

            for row in distinct_attribute_query.all():
                if row.name in attribute_types:
                    if attribute_types[row.name] != row.data_type:
                        # TODO: add runtime resolution of problem
                        #        option: always use scalars where available
                        RuntimeWarning(f'Attribute "{row.name}" has multiple data types in the selected collection. '
                                       f'Using {row.data_type} (not {attribute_types[row.name]}.')

                attribute_types[row.name] = row.data_type

            # Construct query to fetch attributes
            #  Build cases which return correct value field based on attr_name's data_type
            cases = []
            for n in names:
                # Get data type
                data_type = attribute_types[n]

                # Leave out blob format substring
                if data_type.startswith('blob'):
                    if data_type.startswith('blob_ext'):
                        cases.append(
                            sqlalchemy.func.max(sqlalchemy.case(
                                (AttributeTable.name == n, getattr(AttributeTable, f'value_{data_type}')),
                                else_=None)).label(n)
                        )

                        continue

                # Use the appropriate column for the data_type
                cases.append(
                    sqlalchemy.func.max(sqlalchemy.case(
                        (AttributeTable.name == n, getattr(AttributeTable, f'value_{data_type}')),
                        else_=None)).label(n)
                )

            # Construct query
            attribute_query = (
                session.query(
                    EntityTable.uuid,
                    *cases
                )
                .join(AttributeTable, EntityTable.uuid == AttributeTable.entity_uuid)
                .filter(EntityTable.uuid.in_(entity_query.subquery().primary_key))
                .group_by(EntityTable.uuid, EntityTable.id)
            )

        # Create DataFrame from query result
        df = pd.DataFrame(columns=['uuid', *names], data=attribute_query.all())

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
                    df[n] = pd.to_datetime(df[n].apply(lambda s: s.decode()), format='%Y-%m-%d')
                elif data_type == 'datetime':
                    df[n] = pd.to_datetime(df[n].apply(lambda s: s.decode()), format='%Y-%m-%d %H:%M:%S')
                # Load blobs
                elif data_type == 'blob':
                    df[n] = df[n].apply(lambda s: _deserialize(s, _collection.entarchy) if s is not None else None)

            except ValueError:
                raise RuntimeWarning(f'Failed to cast attribute {n} to type {data_type}')

        # Set row index to primary key
        df.set_index('uuid', drop=True, inplace=True)

        return df

    @retry_on_operational_failure
    def set_collection_attributes(self, _collection: Collection, df: pd.DataFrame) -> None:

        _entarchy = _collection.entarchy

        _dtypes = ['str', 'float', 'int', 'date', 'datetime', 'bool', 'blob']

        _analysis_uuid = None
        if _entarchy.current_analysis is not None:
            _analysis_uuid = _entarchy.current_analysis.uuid

        # Do an upsert for each attribute to be updated
        for attr_name in df.columns:

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
            else:
                data_type_str = 'blob'

            # Rename value column
            df_insert.rename(columns={attr_name: f'value_{data_type_str}'}, inplace=True)

            # Add PK set
            df_insert['entity_uuid'] = df_insert.index
            df_insert['name'] = attr_name
            df_insert['data_type'] = data_type_str
            df_insert['analysis_uuid'] = _analysis_uuid

            if not self.db_triggers_enabled:
                df_insert['modified'] = datetime.datetime.now()

            # Perform upsert
            with self.sql_session as session:
                insert_attr_data = df_insert.to_dict('records')
                insert_stmt = sqlalchemy.dialects.mysql.insert(AttributeTable).values(insert_attr_data)
                update_attr_data = {
                    f'value_{data_type_str}': getattr(insert_stmt.inserted, f'value_{data_type_str}'),
                    # On update, reset all other value fields to None:
                    **{f'value_{dt}': None for dt in list(set(_dtypes) - {data_type_str})},
                    'data_type': data_type_str,
                    'analysis_uuid': insert_stmt.inserted.analysis_uuid
                }

                # Add modified time update if triggers are not enabled
                if not self.db_triggers_enabled:
                    update_attr_data['modified'] = datetime.datetime.now()

                # Execute upsert
                upsert_stmt = insert_stmt.on_duplicate_key_update(update_attr_data)
                session.execute(upsert_stmt)
                session.commit()

    def open(self):
        # Just access the property to create the engine if it doesn't exist yet
        # _ = self.sql_engine
        pass

    def close(self):
        self.sql_session.close()
        self.sql_engine.dispose()
        del self._sql_session
        del self._sql_engine
