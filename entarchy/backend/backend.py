from __future__ import annotations

import datetime
from typing import Any, TYPE_CHECKING, Union, Iterable

import pandas as pd

if TYPE_CHECKING:
    from ..core.entarchy import Entarchy
    from ..core.entity import Analysis, Entity, Collection


class Backend(object):
    _config: dict[str, Any] = None
    _debug: bool = False

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    def get_config(self) -> dict[str, Any]:
        return self._config.copy() if self._config is not None else {}

    def close(self):
        raise NotImplementedError('')

    def create(self) -> bool:
        raise NotImplementedError('')

    def create_type_hierarchy(self, _hierarchy: dict[str, ...]) -> bool:
        raise NotImplementedError('')

    def delete(self, confirm: bool = False):
        return NotImplementedError('')

    def open(self):
        raise NotImplementedError('')

    # Entity related methods

    def add_entity(self, _entities: Entity) -> bool:
        raise NotImplementedError('')

    def add_entities(self, _entities: list[Entity]) -> bool:
        raise NotImplementedError('')

    def get_entity_attributes(self, _entity: Entity, names: list[str]) -> tuple[Any, ...]:
        raise NotImplementedError('')

    def get_entity_attribute(self, _entity: Entity, name: str) -> Any:
        raise NotImplementedError('')

    def get_entity_by_uuid(self, entity_uuid: str) -> tuple[str, str, str]:
        raise NotImplementedError('')

    def get_entity_modified_time(self, _entity: Entity) -> datetime.datetime:
        raise NotImplementedError('')

    def get_entities_of_type(self, entity_type: str) -> list[tuple[str, str]]:
        raise NotImplementedError('')

    def get_entity_parent(self, _entity: Entity) -> Union[tuple[str, str, str], None]:
        raise NotImplementedError('')

    def has_entity_attribute(self, _entity: Entity, name: str) -> bool:
        raise NotImplementedError('')

    def set_entity_attribute(self, _entity: Entity, name: str, value: Any):
        raise NotImplementedError('')

    def set_entity_attributes(self, _entity: Entity, names: Iterable[str], value: Iterable[Any]):
        raise NotImplementedError('')

    # Collection related methods

    def get_collection_count(self, _collection: Collection) -> int:
        raise NotImplementedError('')

    def get_collection_entity_by_index(self, _collection: Collection, index: int) -> tuple[str, str]:
        raise NotImplementedError('')

    def get_collection_entity_by_uuid(self, _collection: Collection, index: int) -> tuple[str, str]:
        raise NotImplementedError('')

    def get_collection_entities_by_slice(self, _collection: Collection, _slice: slice) -> list[tuple[str, str]]:
        raise NotImplementedError('')

    def get_collection_attributes(self, _collection: Collection, names: list[str]) -> pd.DataFrame:
        raise NotImplementedError('')

    def get_collection_parent_uuids(self, _collection: Collection) -> list[tuple[str, str]]:
        raise NotImplementedError('')

    def set_collection_attributes(self, _collection: Collection, df: pd.DataFrame) -> None:
        raise NotImplementedError('')
