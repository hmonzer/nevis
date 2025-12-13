import abc
from typing import Generic, TypeVar


TModel = TypeVar("TModel")
TEntity = TypeVar("TEntity")


class BaseEntityMapper(abc.ABC, Generic[TModel, TEntity]):

    @staticmethod
    @abc.abstractmethod
    def to_entity(model_instance: TModel) -> TEntity:
        pass

    @staticmethod
    @abc.abstractmethod
    def to_model(entity: TEntity) -> TModel:
        pass