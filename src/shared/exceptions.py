"""Custom exceptions for the application."""
from typing import Any


class EntityNotFound(Exception):
    """Raised when an entity is not found in the database."""

    def __init__(self, entity_name: str, entity_id: Any):
        """
        Initialize the exception.

        Args:
            entity_name: Name of the entity that was not found
            entity_id: ID of the entity that was not found
        """
        super().__init__(f"{entity_name} with ID {entity_id} not found")
        self.entity_name = entity_name
        self.entity_id = entity_id


class ConflictingEntityFound(Exception):
    """Raised when an entity with a conflicting field already exists."""

    def __init__(self, entity_name: str, field_name: str, field_value: Any):
        """
        Initialize the exception.

        Args:
            entity_name: Name of the entity
            field_name: Name of the conflicting field
            field_value: Value of the conflicting field
        """
        super().__init__(f"{entity_name} with {field_name} '{field_value}' already exists")
        self.entity_name = entity_name
        self.field_name = field_name
        self.field_value = field_value