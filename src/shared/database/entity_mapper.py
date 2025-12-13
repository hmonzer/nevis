from typing import Dict, Type, Callable, Any


class EntityMapper:
    def __init__(self, entity_mappings: Dict[Type, Callable[[Any], Any]]):
        self.entity_mappings = entity_mappings

    def map_to_entity(self, model_instance: Any):
        model_type = type(model_instance)
        if model_type in self.entity_mappings:
            return self.entity_mappings[model_type](model_instance)
        else:
            raise ValueError(f"No entity mapping found for model type: {model_type}")
