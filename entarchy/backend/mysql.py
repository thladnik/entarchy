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
        return f"<Attribute({self.name}, {self.entity})>"


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

    def get_entity_data_of_type(self, _entarchy: Entarchy, _analysis: Analysis, entity_type: str) -> list[tuple[str, str]]:
        with sqlalchemy.orm.Session(self.sql_engine) as session:

            query = (session.query(EntityTable)
                     .join(EntityTypeTable)
                     .filter(EntityTypeTable.name == entity_type)
                     .order_by(getattr(EntityTable.uuid, 'asc')()))

            return [(row.uuid, row.id) for row in query.all()]

    def get_multiple_attributes_of_entity(self, _entarchy: Entarchy, _analysis: Analysis, _uuid: str, names: list[str]):

        with sqlalchemy.orm.Session(self.sql_engine) as session:

            query = session.query(AttributeTable).filter(AttributeTable.entity_uuid == _uuid)
            conditions = []
            for n in names:
                conditions.append(AttributeTable.name == n)

            query = query.filter(sqlalchemy.or_(*conditions))

            rows = {row.name: row for row in query.all()}

        # Read values in order of names
        values = []
        for n in names:
            if n not in rows:
                raise AttributeError(f'Attribute "{n}" not found for entity with UUID {_uuid}.')
            values.append(self._read_data_from_attribute_row(rows[n]))

        return values

    def get_single_attribute_of_entity(self, _entarchy: Entarchy, _analysis: Analysis, _uuid: str, name: str):
        return self.get_multiple_attributes_of_entity(_entarchy, _analysis, _uuid, [name])[0]

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
                _attr_n = f'value_{row.data_type}'
                print(f'Reset {_attr_n}')
                row.__setattr__(_attr_n, None)

                # Write new value
                self._write_data_to_attribute_row(row, v)

            # Add new attributes
            for n in list(set(names) - set(list(existing_rows.keys()))):
                v = values[names.index(n)]

                # Create new attribute row
                new_row = AttributeTable(entity_uuid=_uuid, name=n)
                session.add(new_row)

                # Write data to row
                self._write_data_to_attribute_row(new_row, v)

            # Commit changes
            session.commit()

            return True, ''

    def set_single_attribute_on_entity(self, _entarchy: Entarchy, _analysis: Analysis, _uuid, key: str, value: Any):

        self.set_multiple_attributes_on_entity(_entarchy, _analysis, _uuid, [key], [value])

        return True, ''

    @staticmethod
    def _read_data_from_attribute_row(row: AttributeTable):

        if row.data_type is None:
            raise ValueError('Attribute data type is None.')

        # Load blob
        if row.data_type == 'blob':
            return pickle.loads(row.value_blob)

        # Otherwise load from this row based on data type
        return getattr(row, f'value_{row.data_type}')

    @staticmethod
    def _write_data_to_attribute_row(row: AttributeTable, data: Any):

        # Get corresponding builtin python scalar type for numpy scalars
        if isinstance(data, np.generic):
            data = data.item()

        # Handle scalars
        if type(data) in (str, float, int, bool, date, datetime):

            # Set value type
            value_type_map = {str: 'str', float: 'float', int: 'int',
                              bool: 'bool', date: 'date', datetime: 'datetime'}
            value_type = value_type_map.get(type(data))

            # Some SQL dialects don't support inf float values
            if value_type == 'float' and math.isinf(data):
                value_type = 'blob'
        else:
            value_type = 'blob'

        # Set value on coresponding column basd on type
        if value_type == 'blob':
            data = pickle.dumps(data)

        row.data_type = value_type
        row.__setattr__(f'value_{value_type}', data)