import unittest
from .common import TestCase
from types import SimpleNamespace
import torch
from margipose.dsntnn import make_gauss

from margipose.models.margipose_model import HeatmapColumn, MargiPoseModel
from margipose.data.skeleton import CanonicalSkeletonDesc


class TestMargiPose(TestCase):
    def test_columns(self):
        norm_col = HeatmapColumn(17, heatmap_space='xy', disable_dilation=True)
        chat_col = HeatmapColumn(17, heatmap_space='zy', disable_dilation=True)
        self.assertEqual(
            sum([p.numel() for p in norm_col.parameters()]),
            sum([p.numel() for p in chat_col.parameters()])
        )

    def test_margipose(self):
        with torch.no_grad():
            in_var = torch.randn(1, 3, 256, 256)
            model = MargiPoseModel(CanonicalSkeletonDesc, n_stages=2)
            out_var = model(in_var)
        self.assertEqual(out_var.size(), torch.Size([1, 17, 3]))

    def test_heatmaps_to_coords(self):
        size = (32, 32)
        sigma = 1
        xy_hm = make_gauss(torch.Tensor([[[-0.5, 0.5]]]), size, sigma, normalize=True)
        zy_hm = make_gauss(torch.Tensor([[[0.1, 0]]]), size, sigma, normalize=True)
        xz_hm = make_gauss(torch.Tensor([[[0, 0.2]]]), size, sigma, normalize=True)
        model = SimpleNamespace(average_xy=False)
        xyz = MargiPoseModel.heatmaps_to_coords(model, xy_hm, zy_hm, xz_hm)
        self.assertEqual(xyz, torch.Tensor([[[-0.5, 0.5, 0.15]]]))
        model = SimpleNamespace(average_xy=True)
        xyz = MargiPoseModel.heatmaps_to_coords(model, xy_hm, zy_hm, xz_hm)
        self.assertEqual(xyz, torch.Tensor([[[-0.25, 0.25, 0.15]]]))


if __name__ == '__main__':
    unittest.main()
