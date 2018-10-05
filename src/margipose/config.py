from sacred import Experiment


def add_config_3d_models(ex: Experiment):
    """Adds Sacred named configs model descriptions."""

    ex.add_named_config('margipose_model', model_desc={
        'type': 'margipose',
        'version': '6.0.0',
        'settings': {
            'n_stages': 4,
            'axis_permutation': True,
            'feature_extractor': 'inceptionv4',
            'pixelwise_loss': 'jsd',
        },
    })
