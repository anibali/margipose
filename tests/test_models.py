import pytest
import torch
from dsntnn import make_gauss
from torch.testing import assert_allclose

from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.models.chatterbox_model import ChatterboxModel
from margipose.models.margipose_model import HeatmapColumn, MargiPoseModel


def test_columns():
    norm_col = HeatmapColumn(17, heatmap_space='xy')
    chat_col = HeatmapColumn(17, heatmap_space='zy')
    n_params_normal = sum([p.numel() for p in chat_col.parameters()])
    n_params_permuted = sum([p.numel() for p in norm_col.parameters()])
    assert n_params_normal == n_params_permuted


@pytest.mark.parametrize('feature_extractor', ['inceptionv4', 'resnet18', 'resnet50'])
def test_margipose(feature_extractor):
    with torch.no_grad():
        in_var = torch.randn(1, 3, 256, 256)
        model = MargiPoseModel(CanonicalSkeletonDesc, n_stages=2, axis_permutation=True,
                               feature_extractor=feature_extractor, pixelwise_loss='jsd')
        out_var = model(in_var)
    assert model.xy_heatmaps[-1].size() == torch.Size([1, 17, 32, 32])
    assert out_var.size() == torch.Size([1, 17, 3])


def test_chatterbox():
    with torch.no_grad():
        in_var = torch.randn(1, 3, 256, 256)
        model = ChatterboxModel(CanonicalSkeletonDesc, pixelwise_loss='jsd')
        out_var = model(in_var)
    assert model.xy_heatmaps[-1].size() == torch.Size([1, 17, 32, 32])
    assert out_var.size() == torch.Size([1, 17, 3])


def test_heatmaps_to_coords():
    size = (32, 32)
    sigma = 1
    xy_hm = make_gauss(torch.Tensor([[[-0.5, 0.5]]]), size, sigma, normalize=True)
    zy_hm = make_gauss(torch.Tensor([[[0.1, 0]]]), size, sigma, normalize=True)
    xz_hm = make_gauss(torch.Tensor([[[0, 0.2]]]), size, sigma, normalize=True)
    xyz = MargiPoseModel.heatmaps_to_coords(xy_hm, zy_hm, xz_hm)
    assert_allclose(xyz, torch.Tensor([[[-0.5, 0.5, 0.15]]]))
