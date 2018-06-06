from semantic_version import Version
from margipose.models import ModelFactory
from margipose.models.margipose_model import MargiPoseModelFactory


class ModelRegistry:
    def __init__(self, factory_classes):
        self.factory_classes = factory_classes

    def factory(self, model_desc) -> ModelFactory:
        type = model_desc['type']
        version = Version(model_desc['version'])
        for factory_class in self.factory_classes:
            if factory_class.match(type, version):
                return factory_class(model_desc)
        raise Exception('no matching model factory for {}@{}'.format(type, version))


model_registry_3d = ModelRegistry([
    MargiPoseModelFactory,
])
