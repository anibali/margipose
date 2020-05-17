import os
import torch
from tempfile import TemporaryDirectory
from torch import onnx

from margipose.data.skeleton import CanonicalSkeletonDesc
from margipose.models.margipose_model import MargiPoseModel


def test_onnx_export():
    dummy_input = torch.randn(1, 3, 256, 256)
    model = MargiPoseModel(CanonicalSkeletonDesc, n_stages=1, axis_permutation=True,
                           feature_extractor='inceptionv4', pixelwise_loss='jsd')
    model.eval()
    with TemporaryDirectory() as d:
        onnx_file = os.path.join(d, 'model.onnx')
        onnx.export(model, (dummy_input,), onnx_file, verbose=False)
