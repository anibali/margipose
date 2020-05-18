"""
Differentiable DSNT operations for use in PyTorch computation graphs.
"""

from functools import reduce
from operator import mul

import torch
import torch.nn.functional


def _normalized_linspace(length, dtype=None, device=None):
    """Generate a vector with values ranging from -1 to 1.

    Note that the values correspond to the "centre" of each cell, so
    -1 and 1 are always conceptually outside the bounds of the vector.
    For example, if length = 4, the following vector is generated:

    ```text
     [ -0.75, -0.25,  0.25,  0.75 ]
     ^              ^             ^
    -1              0             1
    ```

    Args:
        length: The length of the vector
        dtype: Data type of vector elements
        device: Device to store vector on

    Returns:
        The generated vector
    """
    if isinstance(length, torch.Tensor):
        length = length.to(device, dtype)
    first = -(length - 1.0) / length
    return torch.arange(length, dtype=dtype, device=device) * (2.0 / length) + first


def _coord_expectation(heatmaps, dim, transform=None):
    """Calculate the coordinate expected value along an axis.

    Args:
        heatmaps: Normalized heatmaps (probabilities)
        dim: Dimension of the coordinate axis
        transform: Coordinate transformation function, defaults to identity

    Returns:
        The coordinate expected value, `E[transform(X)]`
    """

    dim_size = heatmaps.size()[dim]
    own_coords = _normalized_linspace(dim_size, dtype=heatmaps.dtype, device=heatmaps.device)
    if transform:
        own_coords = transform(own_coords)
    summed = heatmaps.view(-1, *heatmaps.size()[2:])
    for i in range(2 - heatmaps.dim(), 0):
        if i != dim:
            summed = summed.sum(i, keepdim=True)
    summed = summed.view(summed.size(0), -1)
    expectations = summed.mul(own_coords.view(-1, own_coords.size(-1))).sum(-1)
    expectations = expectations.view(*heatmaps.size()[:2])
    return expectations


def _coord_variance(heatmaps, dim):
    """Calculate the coordinate variance along an axis.

    Args:
        heatmaps: Normalized heatmaps (probabilities)
        dim: Dimension of the coordinate axis

    Returns:
        The coordinate variance, `Var[X] =  E[(X - E[x])^2]`
    """

    # mu_x = E[X]
    mu_x = _coord_expectation(heatmaps, dim)
    # var_x = E[(X - mu_x)^2]
    var_x = _coord_expectation(heatmaps, dim, transform=lambda x: (x - mu_x) ** 2)

    return var_x


def dsnt(heatmaps):
    """Differentiable spatial to numerical transform.

    Args:
        heatmaps (torch.Tensor): Spatial representation of locations

    Returns:
        Numerical coordinates corresponding to the locations in the heatmaps.
    """

    dim_range = range(-1, 1 - heatmaps.dim(), -1)
    mu = torch.stack([_coord_expectation(heatmaps, dim) for dim in dim_range], -1)
    return mu


def average_loss(losses, mask=None):
    """Calculate the average of per-location losses.

    Args:
        losses (Tensor): Predictions (B x L)
        mask (Tensor, optional): Mask of points to include in the loss calculation
            (B x L), defaults to including everything
    """

    if mask is not None:
        assert mask.size() == losses.size(), 'mask must be the same size as losses'
        losses = losses * mask
        denom = mask.sum()
    else:
        denom = losses.numel()

    # Prevent division by zero
    if isinstance(denom, int):
        denom = max(denom, 1)
    else:
        denom = denom.clamp(1)

    return losses.sum() / denom


def flat_softmax(inp):
    """Compute the softmax with all but the first two tensor dimensions combined."""

    orig_size = inp.size()
    flat = inp.view(-1, reduce(mul, orig_size[2:]))
    flat = torch.nn.functional.softmax(flat, -1)
    return flat.view(*orig_size)


def euclidean_losses(actual, target):
    """Calculate the average Euclidean loss for multi-point samples.

    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).

    Args:
        actual (Tensor): Predictions (B x L x D)
        target (Tensor): Ground truth target (B x L x D)
    """

    assert actual.size() == target.size(), 'input tensors must have the same size'

    # Calculate Euclidean distances between actual and target locations
    diff = actual - target
    dist_sq = diff.pow(2).sum(-1, keepdim=False)
    dist = dist_sq.sqrt()
    return dist


def make_gauss(means, size, sigma, normalize=True):
    """Draw Gaussians.

    This function is differential with respect to means.

    Note on ordering: `size` expects [..., depth, height, width], whereas
    `means` expects x, y, z, ...

    Args:
        means: coordinates containing the Gaussian means (units: normalized coordinates)
        size: size of the generated images (units: pixels)
        sigma: standard deviation of the Gaussian (units: pixels)
        normalize: when set to True, the returned Gaussians will be normalized
    """

    dim_range = range(-1, -(len(size) + 1), -1)
    coords_list = [_normalized_linspace(s, dtype=means.dtype, device=means.device)
                   for s in reversed(size)]

    # PDF = exp(-(x - \mu)^2 / (2 \sigma^2))

    # dists <- (x - \mu)^2
    dists = [(x - mean) ** 2 for x, mean in zip(coords_list, means.split(1, -1))]

    # ks <- -1 / (2 \sigma^2)
    stddevs = [2 * sigma / s for s in reversed(size)]
    ks = [-0.5 * (1 / stddev) ** 2 for stddev in stddevs]

    exps = [(dist * k).exp() for k, dist in zip(ks, dists)]

    # Combine dimensions of the Gaussian
    gauss = reduce(mul, [
        reduce(lambda t, d: t.unsqueeze(d), filter(lambda d: d != dim, dim_range), dist)
        for dim, dist in zip(dim_range, exps)
    ])

    if not normalize:
        return gauss

    # Normalize the Gaussians
    val_sum = reduce(lambda t, dim: t.sum(dim, keepdim=True), dim_range, gauss) + 1e-24
    return gauss / val_sum


def _kl(p, q, ndims):
    eps = 1e-24
    unsummed_kl = p * ((p + eps).log() - (q + eps).log())
    kl_values = reduce(lambda t, _: t.sum(-1, keepdim=False), range(ndims), unsummed_kl)
    return kl_values


def _js(p, q, ndims):
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, ndims) + 0.5 * _kl(q, m, ndims)


def _divergence_reg_losses(heatmaps, mu_t, sigma_t, divergence):
    ndims = mu_t.size(-1)
    assert heatmaps.dim() == ndims + 2, 'expected heatmaps to be a {}D tensor'.format(ndims + 2)
    assert heatmaps.size()[:-ndims] == mu_t.size()[:-1]

    gauss = make_gauss(mu_t, heatmaps.size()[2:], sigma_t)
    divergences = divergence(heatmaps, gauss, ndims)
    return divergences


def js_reg_losses(heatmaps, mu_t, sigma_t):
    """Calculate Jensen-Shannon divergences between heatmaps and target Gaussians.

    Args:
        heatmaps (torch.Tensor): Heatmaps generated by the model
        mu_t (torch.Tensor): Centers of the target Gaussians (in normalized units)
        sigma_t (float): Standard deviation of the target Gaussians (in pixels)

    Returns:
        Per-location JS divergences.
    """

    return _divergence_reg_losses(heatmaps, mu_t, sigma_t, _js)
