from semantic_version import Version

from .margipose_model import MargiPoseModelFactory
from .chatterbox_model import ChatterboxModelFactory


import torch


MODEL_FACTORIES = [
    MargiPoseModelFactory(),
    ChatterboxModelFactory(),
]


def create_model(model_desc):
    type_name = model_desc['type']
    version = Version(model_desc['version'])

    for factory in MODEL_FACTORIES:
        if factory.is_for(type_name, version):
            model = factory.create(model_desc)
            break
    else:
        raise Exception('unrecognised model {} v{}'.format(type_name, str(version)))

    return model


def load_model(model_file):
    details = torch.load(model_file)
    model = create_model(details['model_desc'])
    model.load_state_dict(details['state_dict'])
    return model
