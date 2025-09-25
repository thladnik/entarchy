from __future__ import annotations

import datetime
from typing import Any, TYPE_CHECKING, Union

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

    def create(self) -> bool:
        raise NotImplementedError('')

    def create_type_hierarchy(self, _hierarchy: dict[str, ...]) -> bool:
        raise NotImplementedError('')

    def delete(self, confirm: bool = False):
        return NotImplementedError('')

    # Entity related methods

    def add_entities(self, _entities: list[Entity]) -> bool:
        raise NotImplementedError('')

    def get_entity_by_uuid(self, _entity: Entity) -> tuple[str, str]:
        raise NotImplementedError('')

    def get_entity_last_time_modified(self, _entity: Entity) -> datetime.datetime:
        raise NotImplementedError('')

    def get_entity_of_type(self, entity_type: str) -> list[tuple[str, str]]:
        raise NotImplementedError('')

    def get_multiple_attributes_of_entity(self, _entity: Entity, names: list[str]) -> tuple[Any, ...]:
        raise NotImplementedError('')

    def get_single_attribute_of_entity(self, _entity: Entity, name: str) -> Any:
        raise NotImplementedError('')

    def set_multiple_attributes_on_entity(self, _entity: Entity, names: list[str], value: list[Any]):
        raise NotImplementedError('')

    def set_single_attribute_on_entity(self, _entity: Entity, name: str, value: Any):
        raise NotImplementedError('')

    # Collection related methods

    def get_entity_count_of_collection(self, _collection: Collection) -> int:
        raise NotImplementedError('')

    def get_entity_of_collection_by_index(self, _collection: Collection, index: int) -> tuple[str, str]:
        raise NotImplementedError('')

    def get_entity_of_collection_by_slice(self, _collection: Collection, _slice: slice) -> list[tuple[str, str]]:
        raise NotImplementedError('')

    def get_multiple_attributes_of_collection(self, _collection: Collection, names: list[str]) -> pd.DataFrame:
        raise NotImplementedError('')

    def set_multiple_attributes_on_collection(self, _collection: Collection, df: pd.DataFrame) -> None:
        raise NotImplementedError('')
