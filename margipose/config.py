from sacred import Experiment


def add_config_3d_models(ex: Experiment):
    """Adds Sacred named configs model descriptions."""

    ex.add_named_config('margipose_model', model_desc={
        'type': 'margipose',
        'version': '4.2.4',
        'settings': {
            'coord_space': 'ndc',
            'n_stages': 4,
        },
    })
