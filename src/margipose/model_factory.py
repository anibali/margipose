from semantic_version import Version, Spec
from abc import ABC, abstractmethod


class ModelFactory(ABC):
    def __init__(self, model_type, version_spec):
        super().__init__()
        self.model_type = model_type
        self.version_spec = Spec(version_spec)

    def is_for(self, model_type: str, version: Version):
        """Check if this factory is responsible for the given model type and version."""
        return model_type == self.model_type and version in self.version_spec

    @abstractmethod
    def create(self, model_desc: dict):
        assert self.is_for(model_desc['type'], model_desc['version']),\
               'model_desc does not match this factory'
