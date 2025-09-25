import io
import math
import operator
import pickle
import datetime
from typing import Any, List, Union

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import Index, ForeignKey, String, create_engine, BigInteger, Double
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .backend import Backend
from .. import Analysis, Collection, Entarchy, Entity


class Base(DeclarativeBase):
    pass


class EntityTypeTable(Base):
    __tablename__ = 'entity_types'

    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    parent_pk: Mapped[int] = mapped_column(ForeignKey('entity_types.pk'), nullable=True)
    parent: Mapped['EntityTypeTable'] = relationship('EntityTypeTable', back_populates='children', remote_side=[pk])
    children: Mapped[List['EntityTypeTable']] = relationship('EntityTypeTable', back_populates='parent',
                                                             remote_side=[parent_pk])

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

    created: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.utcnow)
    modified: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    __table_args__ = (
        Index('ix_unique_id_per_parent_uuid', 'parent_uuid', 'id', unique=True),
    )

    def __repr__(self):
        return f"<{self.entity_type.name}Row(id={self.id}, parent={self.parent})>"


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

    created: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.utcnow)
    modified: Mapped[datetime.datetime] = mapped_column(default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    __table_args__ = (
        Index('ix_unique_name_per_entity_uuid', 'entity_uuid', 'name', unique=True),
    )

    def __repr__(self):
        return f"<Attribute({self.name}, {self.entity})>"


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

        # Build the subquery to filter entities matching the comparison
        subquery = (_session.query(AttributeTable.entity_uuid)
                    .filter(AttributeTable.name == name, comparison)
                    .join(EntityTable)
                    .join(EntityTypeTable).filter(EntityTypeTable.name == entity_type_name)
                    .subquery())

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


def _read_data_from_attribute_row(row: AttributeTable):
    if row.data_type is None:
        raise ValueError('Attribute data type is None.')

    # Load blob
    if row.data_type.startswith('blob'):
        _, _format = row.data_type.split('::')

        return _deserialize(row.value_blob, _format)

    # Otherwise load from this row based on data type
    return getattr(row, f'value_{row.data_type}')


def _write_data_to_attribute_row(row: AttributeTable, data: Any):

    # TODO: in future version, information about data type byte number should be included in data_type column
    #  as part of the _format substring
    #  This way the exact data type can be restored upon read (e.g. int8, int16, float32, float64, etc.)
    #  This would mean that python native scalars may be stored as regular 64bit,
    #  while numpy scalars get variable sizes.
    #  This won't affect actual storage though, as the database will use the same column types (bigint, double) anyway.

    # Get corresponding builtin python scalar type for numpy scalars
    if isinstance(data, np.generic):
        data = data.item()

    # Handle scalars and datetime values
    if type(data) in (str, float, int, bool, datetime.date, datetime.datetime):

        # Set value type
        data_type_map = {str: 'str', float: 'float', int: 'int',
                         bool: 'bool', datetime.date: 'date', datetime.datetime: 'datetime'}
        data_type = data_type_map.get(type(data))

        # Some SQL dialects don't support inf float values
        if data_type == 'float' and math.isinf(data):
            data_type = 'blob'
    else:
        data_type = 'blob'

    # Set (potential) previous value to None
    row.__setattr__(f'value_{row.data_type}', None)

    # Set value on corresponding column based on type
    if data_type == 'blob':
        data, _format = _serialize(data)
        row.data_type = f'{data_type}::{_format}'
    else:
        row.data_type = data_type

    # Set data
    row.__setattr__(f'value_{data_type}', data)


def _serialize(data: Any) -> tuple[bytes, str]:
    if isinstance(data, np.ndarray):
        _format = 'npy'
        with io.BytesIO() as buffer:
            np.lib.format.write_array(buffer, data)
            buffer.seek(0)
            _bytes = buffer.read()
    else:
        _format = 'pickle'
        _bytes = pickle.dumps(data)

    return _bytes, _format


def _deserialize(data: bytes, _format: str) -> Any:
    if _format == 'npy':
        with io.BytesIO(data) as buffer:
            buffer.seek(0)
            return np.lib.format.read_array(buffer)
    elif _format == 'pickle':
        return pickle.loads(data)
    else:
        raise ValueError(f'Unknown blob format "{_format}".')


class MySQLBackend(Backend):
    _sql_engine: Union[sqlalchemy.Engine, None] = None
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
                        sqlalchemy.text(f'''
                            SELECT TRIGGER_NAME
                            FROM information_schema.TRIGGERS
                            WHERE TRIGGER_SCHEMA = '{self.dbname}'
                              AND TRIGGER_NAME = 'attributes_touch_entities_ai'
                        ''')
                    )
                    self._db_triggers_enabled = len(res.fetchall()) > 0

        return self._db_triggers_enabled

    @property
    def sql_engine(self):
        if self._sql_engine is None:
            self._sql_engine = create_engine(f'mysql+pymysql://'
                                             f'{self.dbuser}:{self.dbpassword}'
                                             f'@{self.dbhost}/{self.dbname}',
                                             echo=self.debug)


        return self._sql_engine

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
                connection.execute(
                    sqlalchemy.text("""
                        CREATE TRIGGER attributes_touch_entities_ai
                        AFTER INSERT ON attributes FOR EACH ROW
                          UPDATE entities
                          SET modified = UTC_TIMESTAMP(6)
                          WHERE entities.uuid = NEW.entity_uuid
                    """)
                )
                connection.execute(
                    sqlalchemy.text("""
                        CREATE TRIGGER attributes_touch_entities_au
                        AFTER UPDATE ON attributes FOR EACH ROW
                          UPDATE entities
                          SET modified = UTC_TIMESTAMP(6)
                          WHERE entities.uuid = NEW.entity_uuid
                    """)
                )
                connection.execute(
                    sqlalchemy.text("""
                        CREATE TRIGGER attributes_touch_entities_ad
                        AFTER DELETE ON attributes FOR EACH ROW
                          UPDATE entities
                          SET modified = UTC_TIMESTAMP(6)
                          WHERE entities.uuid = OLD.entity_uuid
                                """)
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

    def create_type_hierarchy(self, _hierarchy: dict[str, ...]) -> bool:

        with sqlalchemy.orm.Session(self.sql_engine) as session:
            def _create_entity_type(_hierarchy: dict[str, ...], parent_row: Union[EntityTypeTable, None]):
                for name, children in _hierarchy.items():
                    row = EntityTypeTable(name=name, parent=parent_row)
                    session.add(row)
                    _create_entity_type(children, row)

            # Add custom types
            _create_entity_type(_hierarchy, None)
            session.commit()

        return True

    def delete(self, confirm: bool = False):

        if not confirm:
            return

        print('> Drop schema')
        engine = create_engine(f'mysql+pymysql://{self.dbuser}:{self.dbpassword}@{self.dbhost}')
        with engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'DROP SCHEMA IF EXISTS {self.dbname}'))
            connection.commit()

    # Entity related methods

    def add_entities(self, _entities: list[Entity]) -> bool:

        with sqlalchemy.orm.Session(self.sql_engine) as session:

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

    def get_entity_attributes(self, _entity: Entity, names: list[str]) -> tuple[Any, ...]:

        entity_uuid = _entity.uuid

        with sqlalchemy.orm.Session(self.sql_engine) as session:

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
            values.append(_read_data_from_attribute_row(rows[n]))

        return tuple(values)

    def get_entity_by_uuid(self, entity_uuid) -> tuple[str, str, str]:

        with sqlalchemy.orm.Session(self.sql_engine) as session:

            query = session.query(EntityTable).filter(EntityTable.uuid == entity_uuid)

            count = query.count()
            if count == 0:
                raise KeyError(f'Entity with UUID {entity_uuid} not found in database.')

            row = query.one()

            return row.entity_type.name, row.uuid, row.id

    def get_entity_modified_time(self, _entity: Entity) -> datetime.datetime:

        entity_uuid = _entity.uuid

        with sqlalchemy.orm.Session(self.sql_engine) as session:

            query = session.query(EntityTable.modified).filter(EntityTable.uuid == entity_uuid)

            count = query.count()
            if count == 0:
                raise KeyError(f'Entity with UUID {entity_uuid} not found in database.')

            row = query.one()

        return row.modified

    def get_entities_of_type(self, entity_type: str) -> list[tuple[str, str]]:

        with sqlalchemy.orm.Session(self.sql_engine) as session:
            query = (session.query(EntityTable.uuid, EntityTable.id)
                     .order_by(getattr(EntityTable.uuid, 'asc')()))

            # Filter by entity type if provided
            if entity_type is not None:
                query = (query.join(EntityTypeTable)
                         .filter(EntityTypeTable.name == entity_type))

            return [(row.uuid, row.id) for row in query.all()]

    def get_entity_parent(self, _entity: Entity) -> Union[tuple[str, str, str], None]:
        entity_uuid = _entity.uuid

        with sqlalchemy.orm.Session(self.sql_engine) as session:

            query = session.query(EntityTable).filter(EntityTable.uuid == entity_uuid)

            count = query.count()
            if count == 0:
                return None

            row = query.one()

            if row.parent is None:
                return None
            else:
                return row.parent.entity_type.name, row.parent.uuid, row.parent.id

    def set_entity_attribute(self, entity_uuid, key: str, value: Any):

        self.set_entity_attributes(entity_uuid, [key], [value])

        return True, ''

    def set_entity_attributes(self, _entity: Entity, names: list[str], values: list[Any]) -> tuple[bool, str]:

        _entarchy = _entity.entarchy
        entity_uuid = _entity.uuid

        _analysis_uuid = None
        if _entarchy.current_analysis is not None:
            _analysis_uuid = _entarchy.current_analysis.uuid

        with sqlalchemy.orm.Session(self.sql_engine) as session:

            attribute_query = session.query(AttributeTable).filter(AttributeTable.entity_uuid == entity_uuid)
            conditions = []
            for n in names:
                conditions.append(AttributeTable.name == n)

            attribute_query = attribute_query.filter(sqlalchemy.or_(*conditions))

            existing_rows = {row.name: row for row in attribute_query.all()}

            # Update existing attributes
            for n, row in existing_rows.items():
                v = values[names.index(n)]

                # Write new value
                _write_data_to_attribute_row(row, v)

            # Add new attributes
            for n in list(set(names) - set(list(existing_rows.keys()))):
                v = values[names.index(n)]

                # Create new attribute row
                new_row = AttributeTable(entity_uuid=entity_uuid, name=n, analysis_uuid=_analysis_uuid)
                session.add(new_row)

                # Write data to row
                _write_data_to_attribute_row(new_row, v)

            # Update entity modified time if triggers are not enabled
            if not self.db_triggers_enabled:
                entity_query = session.query(EntityTable).filter(EntityTable.uuid == entity_uuid)
                entity_row = entity_query.one()  # Check that entity exists
                entity_row.modified = datetime.datetime.utcnow()

            # Commit changes
            session.commit()

            return True, ''

    # Collection related methods

    def get_collection_count(self, _collection: Collection, creation_time: datetime.datetime = None) -> int:

        # Fetch result
        with sqlalchemy.orm.Session(self.sql_engine) as session:
            query = _build_query_from_collection(_collection, session)

            res = query.count()

            return res

    def get_collection_entity_by_index(self, _collection: Collection, index: int, creation_time: datetime.datetime = None) -> tuple[str, str]:

        # Fetch result
        with sqlalchemy.orm.Session(self.sql_engine) as session:
            query = _build_query_from_collection(_collection, session)

            res = query.order_by(EntityTable.uuid).offset(index).limit(1).one()

        return res.uuid, res.id

    def get_collection_entities_by_slice(self, _collection: Collection, _slice: slice) -> list[tuple[str, str]]:

        entity_type_name = _collection.entity_type.__name__

        # Calculate indices
        count = self.get_collection_count(_collection, entity_type_name)
        start, stop, step = _slice.indices(count)

        # Fetch result
        with sqlalchemy.orm.Session(self.sql_engine) as session:
            query = _build_query_from_collection(_collection, session)

            res = query.order_by(EntityTable.uuid).offset(start).limit(stop - start).all()

        # TODO: there should be a way to directly query the n-th row using 'ROW_NUMBER() % n'
        #        but it's not clear how is would work in SQLAlchemy ORM; figure out later
        return [(r.uuid, r.id) for r in res[::step]]

    def get_collection_attributes(self, _collection: Collection, names: list[str]) -> pd.DataFrame:

        entity_type_name = _collection.entity_type.__name__

        # Fetch result
        with sqlalchemy.orm.Session(self.sql_engine) as session:

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
                    data_type, _ = data_type.split('::')

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
                elif data_type.startswith('blob'):
                    _, _format = data_type.split('::')
                    df[n] = df[n].apply(lambda s: _deserialize(s, _format) if s is not None else None)

            except ValueError:
                raise RuntimeWarning(f'Failed to cast attribute {n} to type {data_type}')

        # Set row index to primary key
        df.set_index('uuid', drop=True, inplace=True)

        return df

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
                df_insert['modified'] = datetime.datetime.utcnow()

            # Perform upsert
            with sqlalchemy.orm.Session(self.sql_engine) as session:
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
                    update_attr_data['modified'] = datetime.datetime.utcnow()

                # Execute upsert
                upsert_stmt = insert_stmt.on_duplicate_key_update(update_attr_data)
                session.execute(upsert_stmt)
                session.commit()

    def open(self):
        # Just access the property to create the engine if it doesn't exist yet
        _ = self.sql_engine

    def close(self):
        self.sql_engine.dispose()
        self._sql_engine = None
