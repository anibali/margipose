from semantic_version import Version
from abc import ABC, abstractmethod


class ModelFactory(ABC):
    def __init__(self, model_desc: dict):
        super().__init__()
        self.type = model_desc['type']
        self.version = Version(model_desc['version'])
        assert self.match(self.type, self.version), 'model_desc does not match this factory'
        self.settings = self.merge_default_settings(model_desc['settings'])

    def to_model_desc(self):
        return dict(type=self.type, version=str(self.version), settings=self.settings)

    @staticmethod
    @abstractmethod
    def match(type: str, version: Version):
        pass

    @abstractmethod
    def merge_default_settings(self, settings: dict):
        pass

    @abstractmethod
    def build_model(self):
        pass
