import math
import pickle
import pprint
from datetime import date, datetime
from typing import Any, List, Union

import numpy as np
import sqlalchemy
from sqlalchemy import Index, ForeignKey, String, create_engine
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .backend import Backend
from .. import Entarchy, Entity
from ..core.analysis import Analysis


class Base(DeclarativeBase):
    pass


# Entities

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

    __table_args__ = (
        Index('ix_unique_id_per_parent_uuid', 'parent_uuid', 'id', unique=True),
    )

    def __repr__(self):
        return f"<{self.entity_type.name}Row(id={self.id}, parent={self.parent})>"


# Attributes

class AttributeTable(Base):
    __tablename__ = 'attributes'

    entity_uuid: Mapped[str] = mapped_column(String(36), ForeignKey('entities.uuid'), primary_key=True)
    entity: Mapped['EntityTable'] = relationship('EntityTable', back_populates='attributes')

    name: Mapped[str] = mapped_column(String(500), primary_key=True, index=True)

    value_str: Mapped[str] = mapped_column(String(500), nullable=True)
    value_int: Mapped[int] = mapped_column(nullable=True)
    value_float: Mapped[float] = mapped_column(nullable=True)
    value_bool: Mapped[bool] = mapped_column(nullable=True)
    value_date: Mapped[date] = mapped_column(nullable=True)
    value_datetime: Mapped[datetime] = mapped_column(nullable=True)
    value_blob: Mapped[bytes] = mapped_column(LONGBLOB, nullable=True)
    data_type: Mapped[str] = mapped_column(String(20), nullable=True)

    is_persistent: Mapped[bool] = mapped_column(nullable=True)

    __table_args__ = (
        Index('ix_unique_name_per_entity_uuid', 'entity_uuid', 'name', unique=True),
    )

    def __repr__(self):
        return f"<Attribute({self.name}, {self.value}, {self.entity})>"

    @property
    def value(self):

        if self.data_type is None:
            return None

        # If blob, load from associated row in AttributeBlobTable
        if self.data_type == 'blob':
            return pickle.loads(self.value_blob)

        # Otherwise load from this row based on column_str
        return getattr(self, f'value_{self.data_type}')

    @value.setter
    def value(self, value):

        # If blob, dump to associated row in AttributeBlobTable
        if self.data_type == 'blob':
            self.value_blob = pickle.dumps(value)
            return

        # Make NaN's compatible
        if self.data_type == 'float' and np.isnan(value):
            value = None

        # Otherwise write directly
        setattr(self, f'value_{self.data_type}', value)


class MySQLBackend(Backend):

    _sql_engine = None

    def __init__(self, dbname: str, dbhost: str, dbuser: str, dbpassword: str = None, echo: bool = False):

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
            dbuser = input(f'User name for database schema "{dbname}" [default: caload_user]: ')
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
        if self._sql_engine is None:
            self._sql_engine = create_engine(f'mysql+pymysql://'
                                             f'{self.dbuser}:{self.dbpassword}'
                                             f'@{self.dbhost}/{self.dbname}')
        return self._sql_engine

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

    def create(self) -> bool:

        # Create schema
        print(f'> Create database {self.dbname}')
        engine = create_engine(f'mysql+pymysql://{self.dbuser}:{self.dbpassword}@{self.dbhost}')
        with engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'CREATE SCHEMA IF NOT EXISTS {self.dbname}'))
        engine.dispose()

        # Create tables
        print('> Create tables')
        Base.metadata.create_all(self.sql_engine)

        return True

    def create_type_hierarchy(self, _hierarchy: dict[str, ...]) -> bool:

        print('---')
        print('Create entity type hierarchy:')
        pprint.pprint(_hierarchy)
        print('---')

        with sqlalchemy.orm.Session(self.sql_engine) as session:
            def _create_entity_type(_hierarchy: dict[str,  ...], parent_row: Union[EntityTypeTable, None]):
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

    def get_entity_uuids_of_type(self, _entarchy: Entarchy, _analysis: Analysis, entity_type: str) -> list[str]:
        with sqlalchemy.orm.Session(self.sql_engine) as session:

            query = (session.query(EntityTable)
                     .join(EntityTypeTable)
                     .filter(EntityTypeTable.name == entity_type))

            return [row.uuid for row in query.all()]

    def set_multiple_attributes_on_entity(self, _entarchy: Entarchy, _analysis: Analysis, _uuid, names: list[str], values: list[Any]) -> tuple[bool, str]:

        with sqlalchemy.orm.Session(self.sql_engine) as session:

            query = session.query(AttributeTable).filter(AttributeTable.entity_uuid == _uuid)
            conditions = []
            for n in names:
                conditions.append(AttributeTable.name == n)

            query = query.filter(sqlalchemy.or_(*conditions))

            existing_rows = {row.name: row for row in query.all()}

            # Update existing attributes
            for n, row in existing_rows.items():
                v = values[names.index(n)]

                # Set old value to None
                row.__setattr__(f'value_{row.data_type}', None)

                # Write new value
                self._write_value_to_row(row, v)

            # Add new attributes
            for n in list(set(names) - set(list(existing_rows.keys()))):
                v = values[names.index(n)]

                # Create new attribute row
                new_row = AttributeTable(entity_uuid=_uuid, name=n)
                session.add(new_row)

                # Write data to row
                self._write_value_to_row(new_row, v)

            # Commit changes
            session.commit()

            return True, ''

    def _write_value_to_row(self, row: AttributeTable, value: Any):

        # Get corresponding builtin python scalar type for numpy scalars
        if isinstance(value, np.generic):
            value = value.item()

        # Handle scalars
        if type(value) in (str, float, int, bool, date, datetime):

            # Set value type
            value_type_map = {str: 'str', float: 'float', int: 'int',
                              bool: 'bool', date: 'date', datetime: 'datetime'}
            value_type = value_type_map.get(type(value))

            # Some SQL dialects don't support inf float values
            if value_type == 'float' and math.isinf(value):
                value_type = 'blob'
        else:
            value_type = 'blob'

        # Set value on coresponding column basd on type
        if value_type == 'blob':
            value = pickle.dumps(value)

        row.__setattr__(f'value_{value_type}', value)

    def set_single_attribute_on_entity(self, _entarchy: Entarchy, _analysis: Analysis, _uuid, key: str, value: Any):

        self.set_multiple_attributes_on_entity(_entarchy, _analysis, _uuid, [key], [value])

        return True, ''

        attribute_row = None
        pre_data_type_str = ''

        # Get corresponding builtin python scalar type for numpy scalars
        if isinstance(value, np.generic):
            value = value.item()

        # Query attribute row if not in create mode
        if not self.analysis.is_create_mode:
            # Build query
            attribute_query = (self.analysis.session.query(AttributeTable)
                               .filter(AttributeTable.name == key)
                               .filter(AttributeTable.entity_pk == self.row.pk))

            # Evaluate
            if attribute_query.count() == 1:
                attribute_row = attribute_query.one()
                pre_data_type_str = attribute_row.data_type

            elif attribute_query.count() > 1:
                raise ValueError('Wait a minute...')

        # Create row if it doesn't exist yet
        if attribute_row is None:
            attribute_row = AttributeTable(entity=self.row, name=key, is_persistent=self.analysis.is_create_mode)
            self.analysis.session.add(attribute_row)

        # Determine data type of new value

        # Scalars
        if type(value) in (str, float, int, bool, date, datetime):

            # Set value type
            value_type_map = {str: 'str', float: 'float', int: 'int',
                              bool: 'bool', date: 'date', datetime: 'datetime'}
            value_type = value_type_map.get(type(value))

            # Some SQL dialects don't support inf float values
            if value_type == 'float' and math.isinf(value):
                value_type = 'blob'

            # Set column string
            new_data_type_str = value_type

        # Small objects
        # NOTE: there is no universal way to get the byte number of objects
        # Builtin object have __sizeof__(), but this only returns the overhead for some numpy.ndarrays
        # For numpy arrays it's numpy.ndarray.nbytes
        elif (not isinstance(value, np.ndarray) and value.__sizeof__() < self.analysis.max_blob_size) \
                or (isinstance(value, np.ndarray) and value.nbytes < self.analysis.max_blob_size):

            new_data_type_str = 'blob'

        # Large objects or object of unkown type
        else:
            new_data_type_str = 'path'

        # Handle deletion of old values
        if new_data_type_str != pre_data_type_str:
            # Set old value to None
            attribute_row.value = None

        # Handle path types
        if new_data_type_str == 'path':

            # Get previous path (if available)
            data_path = attribute_row.value

            # If no data_path is set yet, generate it
            if data_path is None:
                if isinstance(value, np.ndarray):
                    data_path = f'hdf5:{self.path}/data.hdf5:{key}'
                else:
                    data_path = f'pkl:{self.path}/{key.replace("/", "_")}'

            # Set value to data_path to write to database
            value = data_path

        # Set row type and value
        attribute_row.data_type = new_data_type_str
        attribute_row.value = value
