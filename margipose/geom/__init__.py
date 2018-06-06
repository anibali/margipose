import torch


def normalise_homogeneous(hom):
    """Normalise homogeneous coordinates so that the last component is 1.

    For example, if the coordinates are in 3D space: (x, y, z, w) -> (x/w, y/w, z/w, 1)
    """
    dim = hom.size(-1)
    return hom / hom.narrow(-1, dim - 1, 1)


def cartesian_to_homogeneous(cart):
    hom = torch.cat([cart, torch.ones_like(cart.narrow(-1, 0, 1))], -1)
    return hom


def homogeneous_to_cartesian(hom):
    dim = hom.size(-1)
    cart = hom.narrow(-1, 0, dim - 1) / hom.narrow(-1, dim - 1, 1)
    return cart


def ensure_homogeneous(coords, d):
    if coords.size(-1) == d + 1:
        return coords
    assert coords.size(-1) == d
    return cartesian_to_homogeneous(coords)


def ensure_cartesian(coords, d):
    if coords.size(-1) == d:
        return coords
    assert coords.size(-1) == d + 1
    return homogeneous_to_cartesian(coords)
